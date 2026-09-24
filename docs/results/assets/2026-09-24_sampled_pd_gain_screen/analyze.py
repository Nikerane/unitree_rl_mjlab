"""Reduce all28 gain-screen replays; refuses incomplete or contact-positive data."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root=Path(sys.argv[1])
rows=[]
for gain in range(7):
    fine_summaries=[]; pair_reports=[]; reversals=[]; motion=[]
    for source in (1,3):
        base=root/f"gain-{gain}-source-{source}"
        coarse_dir=base/"dt-0.00025"; fine_dir=base/"dt-0.000125"
        coarse=json.loads((coarse_dir/"summary.json").read_text())
        fine=json.loads((fine_dir/"summary.json").read_text())
        for summary in (coarse,fine):
            assert summary["no_contact"] and summary["contact_worlds"]==0
            assert summary["num_envs"]==128 and summary["pd_updates"]==30
            assert summary["pd_dt"]==.002 and summary["policy_dt"]==.02
            assert summary["horizon_s"]==.06 and summary["learning_updates"]==0
            assert summary["torque_hold_bitwise"] and summary["actual_torque_max_error"]==0
        for key in ("state_hash","command_sha256","runtime_identity","kp","kd","all_actuator_force_range"):
            assert coarse[key]==fine[key], (gain,source,key)
        gap=np.abs(np.asarray(coarse["velocity_peaks"])-fine["velocity_peaks"])
        p95=np.quantile(gap,.95,axis=0); maximum=gap.max(axis=0)
        with np.load(coarse_dir/"substeps.npz") as c, np.load(fine_dir/"substeps.npz") as f:
            assert not np.any(c["contact"]) and not np.any(f["contact"])
            # Both traces are poststep: fine1,3,5... align with coarse0,1,2...
            q_error=c["qpos"]-f["qpos"][1::2]
            v_error=c["velocity"]-f["velocity"][1::2]
            torque=f["actuator_force"][::16]
            reversals.append(np.sum(torque[1:]*torque[:-1]<0,axis=0))
            motion.append(np.linalg.norm(f["head_xyz"][-1]-f["head_xyz"][0],axis=1)*1000)
            pair_reports.append(dict(source=source,
                peak_speed_difference_p95=p95.tolist(),peak_speed_difference_max=maximum.tolist(),
                numerical_budget_pass=bool(np.all(p95<=.05*3.1415) and np.all(maximum<=.10*3.1415)),
                qpos_rms_difference=np.sqrt(np.mean(q_error**2,axis=(0,1))).tolist(),
                arm_velocity_rms_difference=np.sqrt(np.mean(v_error**2,axis=(0,1))).tolist()))
        fine_summaries.append(fine)
    velocity=np.concatenate([s["velocity_peaks"] for s in fine_summaries])
    chatter=np.concatenate([s["max_saturated_reversals"] for s in fine_summaries])
    error=np.concatenate([s["target_error_rms_rad"] for s in fine_summaries])
    saturation=np.concatenate([s["saturation_fraction"] for s in fine_summaries])
    coarse_chatter=np.concatenate([json.loads((root/f"gain-{gain}-source-{source}"/"dt-0.00025"/"summary.json").read_text())["max_saturated_reversals"] for source in (1,3)])
    clean=bool(np.all(chatter<3) and np.all(coarse_chatter<3))
    rows.append(dict(gain_id=gain,kp=fine_summaries[0]["kp"],kd=fine_summaries[0]["kd"],
        velocity_max=velocity.max(axis=0).tolist(),
        speed_violation_counts=(velocity>3.1415).sum(axis=0).tolist(),
        chatter_world_counts=(chatter>=3).sum(axis=0).tolist(),
        chatter_world_counts_coarse=(coarse_chatter>=3).sum(axis=0).tolist(),
        longest_saturated_reversals=chatter.max(axis=0).tolist(),
        total_torque_reversals_median=np.median(np.concatenate(reversals),axis=0).tolist(),
        saturation_fraction=saturation.mean(axis=0).tolist(),
        joint_target_rms_rad=np.sqrt(np.mean(error**2,axis=0)).tolist(),
        head_displacement_mm_median=float(np.median(np.concatenate(motion))),
        startup_candidate=clean and all(p["numerical_budget_pass"] for p in pair_reports),
        pairs=pair_reports))
print(json.dumps(dict(replays=28,worlds_per_setting_at_each_resolution=256,
    source_note="Two old policy tapes; replay worlds are not independent training seeds.",
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    rows=rows),indent=2))
