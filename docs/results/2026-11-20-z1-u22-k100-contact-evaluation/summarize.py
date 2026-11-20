"""Summarize first attempts; use exact native snapshots and signed cap margins."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def summarize(directory, baseline):
    directory=Path(directory)
    trace=torch.load(directory/'tracking.pt',map_location='cpu',weights_only=True)
    original=torch.load(baseline,map_location='cpu',weights_only=True)
    protocol=json.loads((directory/'evaluation-protocol.json').read_text())
    assert protocol['termination']=='contact' and protocol['training_termination']=='reference'
    assert protocol['reference_steps']==21 and protocol['max_episode_steps']==250
    assert protocol['joint_load_window_s']==.05 and protocol['joint_load_subtract_baseline'] is True
    active=torch.stack([r['active'] for r in trace]).numpy().astype(bool)
    T,N=active.shape
    assert N>0 and N==len(original[0]['active']) and len(original)==21 and T>=21
    differences={}
    for key in ('head_m','target_m','error_m','recorded_time_phase'):
        a=torch.stack([r[key] for r in trace[:21]])
        b=torch.stack([r[key] for r in original])
        differences[key]=float((a-b).abs().max())
    prefix_passed=(differences['head_m']<=1e-5 and differences['target_m']<=1e-6
                   and differences['error_m']<=1e-5 and differences['recorded_time_phase']<=1e-6)
    assert active[:21].all()
    def stack(key):return torch.stack([r[key] for r in trace]).numpy()
    final=active.sum(0)-1
    worlds=np.arange(N)
    def last(key):return stack(key)[final,worlds]
    load=stack('joint_load_nms');speed=stack('joint_speed_peak_rad_s')
    load_margin=torch.stack([r['constraint_excess']['joint_impulse_excess'] for r in trace]).numpy()
    speed_margin=torch.stack([r['constraint_excess']['joint_velocity_excess'] for r in trace]).numpy()
    caps=np.asarray(protocol['joint_load_caps_nms'],dtype=np.float32)
    speed_cap=np.float32(protocol['joint_speed_limit_rad_s'])
    np.testing.assert_allclose(load-caps,load_margin,rtol=0,atol=1e-7)
    np.testing.assert_allclose(speed-speed_cap,speed_margin,rtol=0,atol=1e-6)
    assert np.isfinite(load).all() and np.isfinite(speed).all() and (load>=0).all() and (speed>=0).all()
    per_world_load=np.where(active[:,:,None],load,0).max(0)
    per_world_speed=np.where(active[:,:,None],speed,0).max(0)
    load_exceeded=((load_margin>0)&active[:,:,None]).any(0)
    speed_exceeded=((speed_margin>0)&active[:,:,None]).any(0)
    complete=last('impulse_window_complete').astype(bool)
    contact=last('contact').astype(bool)
    impulse=last('delivered_impulse_ns')
    errors=stack('error_m')*1000
    time=stack('time_s')
    def rmse(mask):return float(np.sqrt(np.mean(errors[mask].astype(float)**2))) if mask.any() else None
    episodes=[]
    for world in range(N):
        episodes.append(dict(episode=world,contact=bool(contact[world]),complete_J50=bool(complete[world]),
            contact_time_s=float(last('contact_time_s')[world]) if contact[world] else None,
            end_time_s=float(last('time_s')[world]),J50_ns=float(impulse[world]) if complete[world] else None,
            partial_impulse_ns=float(impulse[world]),productive=bool(last('productive')[world]),
            peak_depth_gain_m=float(last('peak_depth_m')[world]-last('depth_at_contact_m')[world]) if contact[world] else None,
            precontact_axial_speed_m_s=float(last('axial_speed_m_s')[world]) if contact[world] else None,
            peak_joint_load_nms=per_world_load[world].tolist(),peak_joint_speed_rad_s=per_world_speed[world].tolist(),
            load_exceeded=load_exceeded[world].tolist(),speed_exceeded=speed_exceeded[world].tolist()))
    values=impulse[complete].astype(float)
    prefix=np.arange(T)[:,None]<21
    summary=dict(attempts=N,contacts=int(contact.sum()),complete_windows=int(complete.sum()),
        censored_contacts=int((contact&~complete).sum()),productive=int(last('productive').sum()),
        termination_counts={key:int((stack(field)&active).sum()) for key,field in
            [('contact','contact_termination'),('timeout','time_out'),('reference','reference_exhausted')]},
        J50_ns=dict(mean=float(values.mean()) if len(values) else None,
                    median=float(np.median(values)) if len(values) else None,
                    p95=float(np.percentile(values,95)) if len(values) else None),
        full_clip_rmse_mm=rmse(active&prefix),full_episode_rmse_mm=rmse(active),post_reference_rmse_mm=rmse(active&~prefix),
        load_caps_nms=caps.astype(float).tolist(),speed_cap_rad_s=float(speed_cap),
        max_joint_load_nms=per_world_load.max(0).tolist(),max_joint_load_ratio=(per_world_load/caps).max(0).tolist(),
        load_exceeding_attempts_per_joint=load_exceeded.sum(0).tolist(),load_exceeding_attempts=int(load_exceeded.any(1).sum()),
        max_joint_speed_rad_s=per_world_speed.max(0).tolist(),speed_exceeding_attempts_per_joint=speed_exceeded.sum(0).tolist(),
        speed_exceeding_attempts=int(speed_exceeded.any(1).sum()),prefix_max_abs_differences=differences,
        original_prefix_comparison_passed=prefix_passed,episodes=episodes)
    (directory/'measurements.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in summary.items() if k!='episodes'},indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('baseline');a=p.parse_args()
    summarize(a.directory,a.baseline)
