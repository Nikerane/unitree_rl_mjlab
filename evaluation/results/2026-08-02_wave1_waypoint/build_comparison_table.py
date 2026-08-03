"""Join the CPU renderer trajectory metrics to the CPU impulse diagnostic metrics.

BOTH halves are CPU fixed-reset runs of the SAME checkpoint from the SAME realized
reset digest, so the join is physically coherent. The C0-seed3 pilot verified that
every same-phase shared channel is bit-identical between the two tools.
"""

import csv
import hashlib
import json
import numpy as np
from pathlib import Path

ROOT = Path("evaluation/results/2026-08-02_wave1_waypoint")
ORDER = [("C0", 2), ("C0", 3), ("G", 2), ("G", 3), ("P", 2), ("P", 3)]
SHORT = {"C0": "wave1_c0", "G": "wave1_g", "P": "wave1_p"}
QVEL_LIMIT = 3.1415
NAIL_STOP_M = 0.032

man = json.loads((ROOT / "artifact_manifest.json").read_text())
rows_by_id = {(p["arm"], p["seed"]): p for p in man["policies"]}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


out_rows = []
for arm, seed in ORDER:
    leaf = SHORT[arm] + f"_seed{seed}"
    r = dict(np.load(ROOT / "videos" / leaf / "trace.npz"))
    rmeta = json.loads((ROOT / "videos" / leaf / "metadata.json").read_text())
    d = dict(np.load(ROOT / "impulse" / leaf / "trace.npz"))
    dmeta = json.loads((ROOT / "impulse" / leaf / "metadata.json").read_text())

    pos = r["substep_head_position_m"]
    contact = r["substep_contact"].astype(bool)
    entry = r["guideline_entry_m"]
    nail = r["guideline_nail_m"]
    axis = nail - entry
    axis_u = axis / np.linalg.norm(axis)

    # Perpendicular error: ONLY the PRODUCTION-recorded field. Never re-derived.
    # src/tasks/hammer/mdp/guideline.py:project_to_reference() computes it against the
    # FINITE entry->nail SEGMENT (progress clamped to [0,1]).
    rel = pos - entry
    along = rel @ axis_u
    perp = r["substep_perpendicular_error_m"]

    # PRE-CONTACT WINDOW, deterministic: substeps [0 .. first contact-onset], INCLUSIVE.
    # This is the approach path -- the quantity that actually measures path quality.
    # With no contact anywhere, the whole rollout is pre-contact by construction.
    onset = int(np.argmax(contact)) if contact.any() else len(perp) - 1
    perp_pre = perp[: onset + 1]

    # path-length ratio: evaluated over the SAME pre-contact window (reset -> first
    # contact), so it is directly comparable to precontact_segment_perp_*.
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    if contact.any():
        first_c = int(np.argmax(contact))
        path_len = float(seg[:first_c].sum())
        direct = float(np.linalg.norm(pos[first_c] - entry))
        ratio = path_len / direct if direct > 0 else float("nan")
        t_first = first_c * 0.002
    else:
        first_c, path_len, direct, ratio, t_first = -1, float(seg.sum()), 0.0, float("nan"), float("nan")

    # backward progress along the guideline axis (negative advance)
    dalong = np.diff(along)
    backward = float(-dalong[dalong < 0].sum())
    # lateral excursion = |Y component of the residual off the guideline AXIS|
    # (unclamped axis, deliberately: this measures sideways drift, NOT the finite-segment error)
    lateral = float(np.abs(rel - np.outer(along, axis_u))[:, 1].max())

    qvel_post = np.abs(r["substep_arm_qvel_rad_s"])
    peak_qvel = float(qvel_post.max())

    lam = np.abs(d["lambda_windowed_constraint_read_n_m_s"])
    caps = np.array(dmeta["j_limit_n_m_s"], dtype=float)
    lam_peak = lam.max(axis=0)
    lam_ratio = lam_peak / caps

    payouts = r["control_step_reward_terms"]
    names = [str(n) for n in r["control_step_reward_term_names"]]
    treat = {"C0": None, "G": "r_gate", "P": "r_waypoint_progress"}[arm]
    treat_payout = float(payouts[:, names.index(treat)].sum()) if treat else 0.0

    out_rows.append({
        "arm": arm, "training_seed": seed,
        "checkpoint_sha256": rows_by_id[(arm, seed)]["checkpoint_sha256"][:16],
        # --- CPU fixed-reset trajectory (renderer) ---
        "traj_has_contact": bool(contact.any()),
        "traj_contact_substeps": int(contact.sum()),
        "traj_contact_dwell_ms": round(float(contact.sum()) * 2.0, 1),
        "traj_success": rmeta["rollout"]["terminal_boundary"]["reason"] == "terminated",
        "traj_terminal_reason": rmeta["rollout"]["terminal_boundary"]["reason"],
        "traj_control_steps": int(r["executed_control_steps"]),
        "traj_nail_depth_physical_mm": round(min(float(r["substep_nail_depth_m"].max()), NAIL_STOP_M) * 1e3, 3),
        "traj_nail_depth_raw_mm": round(float(r["substep_nail_depth_m"].max()) * 1e3, 3),
        # gates completed: max over the FULL rollout (a gate crossed at any point counts)
        "traj_gates_completed": int(r["substep_gate_index"].max()),
        "traj_first_contact_s": round(t_first, 4) if t_first == t_first else "",
        # PRIMARY path-quality result: approach path, reset -> first contact inclusive.
        "precontact_segment_perp_max_mm": round(float(perp_pre.max()) * 1e3, 2),
        "precontact_segment_perp_rms_mm": round(float(np.sqrt((perp_pre ** 2).mean())) * 1e3, 2),
        # SECONDARY: whole rollout. Inflated by post-strike follow-through past the nail,
        # where finite-segment projection clamps the closest point to the nail endpoint.
        "full_rollout_segment_perp_max_mm": round(float(perp.max()) * 1e3, 2),
        "full_rollout_segment_perp_rms_mm": round(float(np.sqrt((perp ** 2).mean())) * 1e3, 2),
        "precontact_window_substeps": int(onset + 1),
        "traj_path_len_ratio": round(ratio, 4) if ratio == ratio else "",
        "traj_backward_travel_mm": round(backward * 1e3, 2),
        "traj_lateral_excursion_mm": round(lateral * 1e3, 2),
        "traj_peak_qvel_rad_s": round(peak_qvel, 4),
        "traj_qvel_legal": bool(peak_qvel <= QVEL_LIMIT),
        "traj_treatment_payout": round(treat_payout, 6),
        # --- CPU fixed-reset impulse diagnostic ---
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
    })

with (ROOT / "tables" / "wave1_six_policy_comparison.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
    w.writeheader()
    w.writerows(out_rows)

print("=== CPU fixed-reset trajectory (renderer) ===")
hdr = ("arm/seed", "contact", "succ", "steps", "depth_mm", "gates",
       "preP_max", "preP_rms", "fullP_max", "pathR", "qvel", "legal", "payout")
print("  " + "".join(f"{h:>10}" for h in hdr))
for r_ in out_rows:
    print("  " + "".join(f"{v:>10}" for v in (
        f"{r_['arm']}/{r_['training_seed']}", r_["traj_contact_substeps"],
        str(r_["traj_success"])[:5], r_["traj_control_steps"],
        r_["traj_nail_depth_physical_mm"], r_["traj_gates_completed"],
        r_["precontact_segment_perp_max_mm"], r_["precontact_segment_perp_rms_mm"],
        r_["full_rollout_segment_perp_max_mm"],
        r_["traj_path_len_ratio"], r_["traj_peak_qvel_rad_s"],
        str(r_["traj_qvel_legal"])[:5], r_["traj_treatment_payout"])))

print()
print("=== CPU fixed-reset impulse diagnostic ===")
hdr2 = ("arm/seed", "v_pre", "delivered", "F_ax", "L_j1", "L_j2", "L_j5",
        "maxRatio", "joint", "dwell")
print("  " + "".join(f"{h:>11}" for h in hdr2))
for r_ in out_rows:
    print("  " + "".join(f"{v:>11}" for v in (
        f"{r_['arm']}/{r_['training_seed']}", r_["imp_v_precontact_m_s"],
        r_["imp_delivered_n_s"], r_["imp_axial_force_peak_n"],
        r_["imp_lambda_max_j1"], r_["imp_lambda_max_j2"], r_["imp_lambda_max_j5"],
        r_["imp_max_cap_ratio"], r_["imp_max_cap_joint"],
        r_["imp_contact_substeps"])))

print()
print("table sha256:", sha256(ROOT / "tables" / "wave1_six_policy_comparison.csv")[:16])
