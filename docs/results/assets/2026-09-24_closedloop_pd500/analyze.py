"""Reduce first episodes only; completed impact windows cannot cross resets."""
import json
from pathlib import Path
import sys
import numpy as np
from gain_screen import max_saturated_reversals
from impact_window import reduce_window

root=Path(sys.argv[1])
rows=[];summaries=[]


def stats(x):
    return None if not len(x) else dict(median=np.median(x,axis=0).tolist(),
        p95=np.quantile(x,.95,axis=0).tolist(),max=np.max(x,axis=0).tolist())


for arm in range(4):
    base=root/f"arm-{arm}"
    s=json.loads((base/"summary.json").read_text());summaries.append(s)
    assert s["learning_updates"]==0 and s["pd_dt"]==.002 and s["policy_dt"]==.02
    assert s["pd_updates"]==10*s["step_count"]
    assert s["actual_torque_error"]==0 and s["torque_hold_bitwise"]
    dt=s["physics_dt"];stride=round(.002/dt);dec=round(.02/dt)
    with np.load(base/"actions.npz") as a:
        done=a["first_done"].copy()
        end=np.where(done>=0,done,s["step_count"]-1)
    with np.load(base/"substeps.npz") as z,np.load(base/"transitions.npz") as t:
        active=z["active"];tc=t["active"]
        expected=(np.arange(len(active))[:,None]//dec)<=end[None,:]
        assert np.array_equal(active,expected)
        assert np.array_equal(tc,np.arange(s["step_count"])[:,None]<=end[None,:])
        contact=z["contact"]&active
        w=reduce_window(z["contact_row_qfrc"],z["force_axial"],contact,dt,end)
        valid=w["source_complete"]
        velocity=np.where(active[:,:,None],np.abs(z["velocity"]),0).max(axis=0)
        np.testing.assert_array_equal(velocity,np.asarray(s["first_episode_velocity_peak"],dtype=velocity.dtype))
        torque=np.where(active[:,:,None],z["actuator_force"],0)
        caps=np.array([[-30,30],[-60,60],[-30,30],[-30,30],[-30,30],[-30,30]])
        saturated=((torque<=caps[None,None,:,0]+1e-4)|(torque>=caps[None,None,:,1]-1e-4))&active[:,:,None]
        chatter=max_saturated_reversals(torque[::stride],caps)
        measured=t["outcome_valid_measured"]&tc
        measured_worlds=measured.any(axis=0);idx=measured.argmax(axis=0)
        speeds=t["outcome_measured_mps"][idx,np.arange(s["num_envs"])][measured_worlds]
        rows.append(dict(arm=arm,dose=s["dose"],physics_dt=dt,
            first_episodes_done=s["first_episodes_done"],censored=s["censored_first_episodes"],
            contact_worlds=int(w["started"].sum()),complete_impact_windows=int(valid.sum()),
            incomplete_contact_windows=int((w["started"]&~valid).sum()),
            native_productive_worlds=int((t["outcome_productive"]&tc).any(axis=0).sum()),
            native_miss_worlds=int((t["outcome_miss"]&tc).any(axis=0).sum()),
            native_incomplete_worlds=int((t["outcome_incomplete"]&tc).any(axis=0).sum()),
            speed_max=velocity.max(axis=0).tolist(),speed_violations=(velocity>3.1415).sum(axis=0).tolist(),
            saturation_fraction=(saturated.sum(axis=(0,1))/active.sum()).tolist(),
            sustained_chatter_worlds=(chatter>=3).sum(axis=0).tolist(),
            tracking_rms_mm=float(np.sqrt(np.sum(t["path_error_m"]**2*tc)/tc.sum())*1000),
            measured_impact_speed_mps=stats(speeds),
            delivered_complete_Ns=stats(w["delivered"][valid]),
            contact_joint_complete_Nms=stats(w["joint_impulse"][valid])))
for first,second in ((0,1),(2,3)):
    assert summaries[first]["reset_hashes"]==summaries[second]["reset_hashes"]
    assert summaries[first]["checkpoint_sha256"]==summaries[second]["checkpoint_sha256"]
print(json.dumps(dict(rows=rows,paired_initial_evidence_equal=True,learning_updates=0,
    limitations=["Two old policy instances under an explicit fixed-gain controller intervention.",
        "500ms first-episode censoring, not final task success.",
        "Native productive event is not proof of task completion or safety.",
        "Contact-row impulse has no validated hardware safety threshold.",
        "Closed-loop trajectories may diverge between physics resolutions."]),indent=2,allow_nan=False))
