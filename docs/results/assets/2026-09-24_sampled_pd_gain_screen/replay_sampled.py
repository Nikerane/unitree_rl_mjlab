"""Frozen-target 60ms no-contact gain screen: 50Hz targets, 500Hz PD, independent physics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from sampled_pd import SampledPD, write_held_torque
from gain_screen import apply_gains, max_saturated_reversals


def replay(capture, output, dt, kp_scale, arm_kd, device, tools):
    import mujoco
    import mjlab.tasks
    import src.tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from src.tasks.hammer.rl.impact_speed_wrapper import configure_impact_speed_experiment
    from contact_runtime import restore_initial, runtime_identity, preflight, sample, stacked, write_json
    from contact_replay_helpers import duration_steps, command_at, passive_window, require_pair

    meta=json.loads((capture/'capture.json').read_text())
    if runtime_identity() != meta['runtime_identity']:
        raise ValueError('source/dependency identity mismatch')
    state=torch.load(capture/'initial.pt',map_location='cpu',weights_only=False)
    require_pair(meta,state)
    with np.load(capture/'tape.npz') as archive:
        target_tape=archive['target'].copy(); terminal=archive['terminal_steps'].copy()
    decimation=duration_steps(.02,dt); window=duration_steps(.05,dt)
    cfg=load_env_cfg(meta['task'],play=False)
    configure_impact_speed_experiment(cfg,bounds=tuple(meta['speed_bounds']),cost=meta['speed_cost'],mode='external')
    cfg.scene.num_envs=meta['num_envs']; cfg.seed=meta['seed']
    cfg.metrics['cat_soft'].params['imp_max_p']=meta['dose']
    cfg.sim.mujoco.timestep=dt; cfg.decimation=decimation
    cfg.metrics['first_strike'].params['window_substeps']=window
    cfg.metrics['substep_impulse'].params['event_window_substeps']=window
    cfg.metrics['substep_delivered'].params.update(event_window_substeps=window,rearm_gap_substeps=window)
    env=ManagerBasedRlEnv(cfg,device=device)
    try:
        env.seed(meta['seed']); env.reset(); preflight(env); restore_initial(env,state)
        position=env.action_manager.get_term('joint_position')
        gains=env.action_manager.get_term('joint_stiffness')
        model=env.sim.mj_model
        assert model.nu == 7, 'explicitly account for all seven robot actuators'
        assert np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        assert np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
        assert np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
        assert np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
        np.testing.assert_array_equal(model.actuator_gear,np.tile([1.,0,0,0,0,0],(model.nu,1)))
        assert np.all(model.actuator_forcelimited), 'native effort limits must stay active'
        limits=torch.as_tensor(model.actuator_forcerange.copy(),dtype=torch.float32,device=device)
        joints=model.actuator_trnid[:,0]
        qids=torch.as_tensor(model.jnt_qposadr[joints],device=device)
        vids=torch.as_tensor(model.jnt_dofadr[joints],device=device)
        kp=env.sim.model.actuator_gainprm[:,:,0].clone()
        kd=-env.sim.model.actuator_biasprm[:,:,2].clone()
        arm_ids=gains.control_ids
        arm_kp=apply_gains(kp,kd,arm_ids,kp_scale,arm_kd)
        gains._kp[:]=arm_kp; gains._kd[:]=arm_kd
        pd=SampledPD(dt); rows=[]; commands=[]; applied_error=0.
        for step in range(3):
            target=command_at(target_tape,terminal,step)
            position._processed_actions.copy_(torch.as_tensor(target,device=device))
            commands.append(target)
            for substep in range(decimation):
                index=step*decimation+substep
                env._sim_step_counter+=1
                env.action_manager.apply_action(); env.scene.write_data_to_sim()
                torque=pd.update(index,q=env.sim.data.qpos[:,qids],v=env.sim.data.qvel[:,vids],
                    target=env.sim.data.ctrl,kp=kp,kd=kd,limits=limits)
                write_held_torque(env.sim.model,torque)
                env.sim.step(); env.scene.update(dt=dt); env.metrics_manager.compute_substep()
                actual=env.sim.data.actuator_force[:]
                torch.testing.assert_close(actual,torque,rtol=0,atol=1e-5)
                applied_error=max(applied_error,float((actual-torque).abs().max()))
                row=sample(env)
                tracker=env._hammer_first_strike
                head=tracker._robot.data.site_pos_w[:,tracker._site_ids].squeeze(1)
                phase=env._strike_reference.preview(head,env.episode_length_buf)
                ref=env._strike_reference.waypoint(phase)
                row.update(joint_target_error=(env.sim.data.ctrl[:,arm_ids]-env.sim.data.qpos[:,qids[arm_ids]]).detach().cpu().numpy().copy(),
                           held_torque=torque.detach().cpu().numpy().copy(),
                           requested_torque=pd.requested.detach().cpu().numpy().copy(),
                           head_xyz=head.detach().cpu().numpy().copy(),
                           reference_error=(head-ref).detach().cpu().numpy().copy())
                rows.append(row)
            env.episode_length_buf+=1; env.common_step_counter+=1
            env.sim.forward(); env.command_manager.compute(dt=.02); env.sim.sense()
        assert pd.updates == 30, '60ms must contain exactly30 PD updates'
        traces=stacked(rows)
        # Verify torque is bitwise constant over each2ms block, all worlds/all7 actuators.
        blocks=traces['held_torque'].reshape(30,pd.stride,env.num_envs,model.nu)
        assert np.array_equal(blocks,np.repeat(blocks[:,:1],pd.stride,axis=1))
        passive=passive_window(traces['contact'],traces['force_axial'],dt)
        velocity=np.abs(traces['velocity']).max(axis=0)
        torque=traces['held_torque'][:,:,arm_ids.cpu().numpy()]
        cap=limits[arm_ids].cpu().numpy()
        saturation=(torque<=cap[None,None,:,0]+1e-4)|(torque>=cap[None,None,:,1]-1e-4)
        output.mkdir()
        np.savez_compressed(output/'substeps.npz',**traces)
        np.savez_compressed(output/'commands.npz',target=np.stack(commands))
        summary=dict(physics_dt=dt,pd_dt=.002,policy_dt=.02,window_s=.05,horizon_s=.06,
            num_envs=env.num_envs,pd_updates=pd.updates,learning_updates=0,
            kp=arm_kp.cpu().tolist(),kd=[arm_kd]*6,kp_scale=kp_scale,
            no_contact=not bool(np.any(traces["contact"])),
            contact_worlds=int(np.any(traces["contact"],axis=0).sum()),
            max_saturated_reversals=max_saturated_reversals(torque[::pd.stride],cap).tolist(),
            target_error_rms_rad=np.sqrt(np.mean(traces["joint_target_error"]**2,axis=0)).tolist(),
            all_actuator_kp=kp[0].cpu().tolist(),all_actuator_kd=kd[0].cpu().tolist(),
            all_actuator_force_range=limits.cpu().tolist(),
            torque_hold_bitwise=True,actual_torque_max_error=applied_error,
            state_hash=meta['state_hash'],runtime_identity=meta['runtime_identity'],
            command_sha256=hashlib.sha256(np.stack(commands).tobytes()).hexdigest(),
            capture=str(capture),velocity_peaks=velocity.tolist(),
            speed_violation_counts=(velocity>3.1415).sum(axis=0).tolist(),
            saturation_fraction=saturation.mean(axis=0).tolist(),
            tracking_rms_mm=(np.sqrt(np.mean(np.sum(traces['reference_error']**2,axis=2),axis=0))*1000).tolist(),
            passive={k:v.tolist() for k,v in passive.items()},
            limitations=['frozen commands, not closed-loop policy or fresh training',
                'first60ms frozen commands, startup-only result; source-postterminal targets held if needed',
                'ideal body gravity compensation unchanged; not full Z1 firmware model',
                'native continuously evaluated PD has been replaced, not merely refined'])
        write_json(output/'summary.json',summary)
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-tools',type=Path,required=True)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--kp-scale',type=float,choices=(.5,1.),required=True)
    parser.add_argument('--arm-kd',type=float,choices=(25.6,12.8,6.4,3.2),required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--timesteps',type=float,nargs='+',default=[.00025,.000125])
    args=parser.parse_args()
    sys.path.insert(0,str(args.prior_tools))
    args.output.mkdir(parents=True,exist_ok=False)
    dependencies=[Path(__file__),Path(__file__).with_name('sampled_pd.py'),Path(__file__).with_name('gain_screen.py')]+[
        args.prior_tools/name for name in ('contact_runtime.py','contact_replay_helpers.py')]
    manifest={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
    with (args.output/'dependency-hashes.json').open('x') as stream: json.dump(manifest,stream,indent=2)
    for dt in args.timesteps:
        replay(args.capture,args.output/f'dt-{dt:g}',dt,args.kp_scale,args.arm_kd,args.device,args.prior_tools)


if __name__=='__main__': main()
