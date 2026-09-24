"""Offline paired full-strike audit. Inputs: result directory and capture parent."""
import json
from pathlib import Path
import sys
import numpy as np
from impact_window import reduce_window

root=Path(sys.argv[1]); captures=Path(sys.argv[2])
records=[]; comparisons=[]


def stats(values):
    if not len(values):
        return None
    return dict(median=np.median(values,axis=0).tolist(),
        p95=np.quantile(values,.95,axis=0).tolist(),max=np.max(values,axis=0).tolist())


for source_index,source in enumerate((1,3)):
    with np.load(captures/f"contact-{source}-capture/tape.npz") as tape:
        terminal=tape["terminal_steps"].copy()
    paired=[]
    for dt in (.00025,.000125):
        off=root/f"arm-{source_index}"/f"dt-{dt:g}"
        on=root/f"arm-{source_index+2}"/f"dt-{dt:g}"
        summary=json.loads((on/"summary.json").read_text())
        original=json.loads((off/"summary.json").read_text())
        assert summary["observe_contact"] and not original["observe_contact"]
        for s in (summary,original):
            assert s["pd_updates"]==140 and s["pd_dt"]==.002 and s["policy_dt"]==.02
            assert s["horizon_s"]==.28 and s["actual_torque_max_error"]==0
            assert s["torque_hold_bitwise"] and s["learning_updates"]==0
        for key in ("state_hash","command_sha256","runtime_identity","kp","kd"):
            assert summary[key]==original[key]
        for filename in ("substeps.npz","commands.npz"):
            with np.load(off/filename) as a,np.load(on/filename) as b:
                assert set(b.files)-set(a.files)==({"contact_row_qfrc"} if filename=="substeps.npz" else set())
                for key in a.files:
                    assert np.array_equal(a[key],b[key]), (source,dt,filename,key)
        with np.load(on/"substeps.npz") as z:
            contact=z["contact"]; force=z["contact_row_qfrc"]
            windows=reduce_window(force,z["force_axial"],contact,dt,terminal)
            times=(np.arange(len(contact))+1)*dt
            prefix=times[:,None]<=(terminal+1)[None,:]*.02+1e-10
            prefix_speed=np.where(prefix[:,:,None],np.abs(z["velocity"]),0).max(axis=0)
            before_contact=(times[:,None]<windows["onset_s"][None,:])|~windows["started"][None,:]
            pre_speed=np.where(before_contact[:,:,None],np.abs(z["velocity"]),0).max(axis=0)
            whole_speed=np.abs(z["velocity"]).max(axis=0)
            complete=windows["complete"]; source_complete=windows["source_complete"]
            records.append(dict(source=source,dt=dt,additional_recording_noninterference=True,
                reconstruction_max_error=summary["constraint_reconstruction_max_error"],
                contact_count=int(windows["started"].sum()),complete_windows=int(complete.sum()),
                source_prefix_complete_windows=int(source_complete.sum()),
                late_incomplete_windows=int((windows["started"]&~complete).sum()),
                contact_onset_s=stats(windows["onset_s"][windows["started"]]),
                full_speed_max=whole_speed.max(axis=0).tolist(),
                full_speed_violations=(whole_speed>3.1415).sum(axis=0).tolist(),
                source_prefix_speed_max=prefix_speed.max(axis=0).tolist(),
                source_prefix_speed_violations=(prefix_speed>3.1415).sum(axis=0).tolist(),
                before_first_contact_speed_max=pre_speed.max(axis=0).tolist(),
                before_first_contact_speed_violations=(pre_speed>3.1415).sum(axis=0).tolist(),
                chatter_world_counts=(np.asarray(summary["max_saturated_reversals"])>=3).sum(axis=0).tolist(),
                saturation_mean=np.asarray(summary["saturation_fraction"]).mean(axis=0).tolist(),
                delivered_complete_Ns=stats(windows["delivered"][complete]),
                contact_joint_complete_Nms=stats(windows["joint_impulse"][complete]),
                delivered_source_complete_Ns=stats(windows["delivered"][source_complete]),
                contact_joint_source_complete_Nms=stats(windows["joint_impulse"][source_complete]),
                sensor_off_nonzero_contact_rows=int((np.any(np.abs(force)>1e-8,axis=2)&~contact).sum())))
            paired.append(dict(summary=summary,windows=windows,velocity=z["velocity"].copy(),
                qpos=z["qpos"].copy(),whole_speed=whole_speed,prefix_speed=prefix_speed))
    a,b=paired
    for key in ("state_hash","command_sha256","runtime_identity","kp","kd"):
        assert a["summary"][key]==b["summary"][key]
    both=a["windows"]["complete"]&b["windows"]["complete"]
    both_prefix=a["windows"]["source_complete"]&b["windows"]["source_complete"]
    gap=np.abs(a["whole_speed"]-b["whole_speed"])
    p95=np.quantile(gap,.95,axis=0); maximum=gap.max(axis=0)
    comparisons.append(dict(source=source,
        paired_complete_count=int(both.sum()),paired_source_complete_count=int(both_prefix.sum()),
        contact_status_mismatches=int(np.sum(a["windows"]["started"]!=b["windows"]["started"])),
        peak_speed_difference_p95=p95.tolist(),peak_speed_difference_max=maximum.tolist(),
        speed_numerical_screen_pass=bool(np.all(p95<=.05*3.1415) and np.all(maximum<=.10*3.1415)),
        prefix_peak_speed_difference_max=np.abs(a["prefix_speed"]-b["prefix_speed"]).max(axis=0).tolist(),
        qpos_rms_difference=np.sqrt(np.mean((a["qpos"]-b["qpos"][1::2])**2,axis=(0,1))).tolist(),
        velocity_rms_difference=np.sqrt(np.mean((a["velocity"]-b["velocity"][1::2])**2,axis=(0,1))).tolist(),
        onset_difference_s=stats(np.abs(a["windows"]["onset_s"][both]-b["windows"]["onset_s"][both])),
        delivered_difference_Ns=stats(np.abs(a["windows"]["delivered"][both]-b["windows"]["delivered"][both])),
        contact_joint_difference_Nms=stats(np.abs(a["windows"]["joint_impulse"][both]-b["windows"]["joint_impulse"][both])),
        source_complete_delivered_difference_Ns=stats(np.abs(a["windows"]["delivered"][both_prefix]-b["windows"]["delivered"][both_prefix])),
        source_complete_contact_joint_difference_Nms=stats(np.abs(a["windows"]["joint_impulse"][both_prefix]-b["windows"]["joint_impulse"][both_prefix]))))

print(json.dumps(dict(replays=8,learning_updates=0,records=records,comparisons=comparisons,
    limitations=["Frozen old commands, not closed-loop policies.",
        "Full horizon includes post-source-termination holds.",
        "Contact generalized impulse is not internal gearbox load.",
        "No old joint-load safety threshold transferred.",
        "Observer parity concerns additional recording only."]),indent=2,allow_nan=False))
