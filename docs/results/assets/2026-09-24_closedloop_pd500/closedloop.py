"""Closed-loop posttraining controller intervention; no checkpoint reinterpretation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from gain_screen import apply_gains,validate_arm_columns
from hold_step import HeldTorqueStep


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture",type=Path,required=True)
    parser.add_argument("--prior-tools",type=Path,required=True)
    parser.add_argument("--dt",type=float,choices=(.00025,.000125),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.prior_tools))
    import mujoco
    import mjlab.tasks
    import src.tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
    from tensordict import TensorDict
    from contact_runtime import runtime_identity,preflight,sample,stacked
    from contact_replay_helpers import duration_steps
    from scripts.evaluate_impact_speed import PreResetSnapshot,override_latched_goal,_reset_evidence
    from src.tasks.hammer.rl.impact_speed_wrapper import configure_impact_speed_experiment,wrap_impact_speed_env
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata
    from src.tasks.hammer.mdp.contact_row_impulse import contact_row_qfrc,arm_dof_cols

    meta=json.loads((args.capture/"capture.json").read_text())
    identity=runtime_identity()
    assert identity==meta["runtime_identity"]
    checkpoint=Path(meta["checkpoint"])
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected={.2:"b58f539c1439c550e7a5663a5b0f9f82dbabc73a1c9b55a5fb0db0210d01a14f",
              .3:"427aa85a325b79b7fc345e0f79941e98e5fa27b31b49a7d3a15654e29e66b69c"}
    assert checkpoint_hash==expected[meta["dose"]]

    def config():
        cfg=load_env_cfg(meta["task"],play=False)
        configure_impact_speed_experiment(cfg,bounds=tuple(meta["speed_bounds"]),
            cost=meta["speed_cost"],mode="external")
        cfg.scene.num_envs=meta["num_envs"];cfg.seed=meta["seed"]
        cfg.metrics["cat_soft"].params["imp_max_p"]=meta["dose"]
        return cfg

    agent_cfg=load_rl_cfg(meta["task"])
    original=ManagerBasedRlEnv(config(),device="cpu")
    try:
        original_wrapped=wrap_impact_speed_env(original,agent_cfg.clip_actions)
        runner=load_runner_cls(meta["task"])(original_wrapped,asdict(agent_cfg),device="cpu")
        runner.load(str(checkpoint),load_cfg={"actor":True},strict=True,map_location="cpu")
        original_contract=build_impact_speed_metadata(original,original_wrapped.clip_actions)
        policy=runner.get_inference_policy(device="cpu")
    finally:
        original.close()
    # New env is explicitly an intervention. Do not construct a new runner or
    # pretend its different physics/controller satisfy the old checkpoint contract.
    cfg=config();cfg.sim.mujoco.timestep=args.dt
    cfg.decimation=duration_steps(.02,args.dt)
    window=duration_steps(.05,args.dt)
    cfg.metrics["first_strike"].params["window_substeps"]=window
    cfg.metrics["substep_impulse"].params["event_window_substeps"]=window
    cfg.metrics["substep_delivered"].params.update(event_window_substeps=window,rearm_gap_substeps=window)
    args.output.mkdir(parents=True,exist_ok=False)
    env=ManagerBasedRlEnv(cfg,device="cpu")
    try:
        wrapped=wrap_impact_speed_env(env,agent_cfg.clip_actions)
        env.seed(meta["seed"]);obs,_=wrapped.reset();preflight(env)
        reset_hashes=_reset_evidence(env,obs)
        obs=TensorDict(override_latched_goal(env,meta["goal"]),batch_size=[env.num_envs])
        model=env.sim.mj_model
        assert model.nu==7
        assert np.all(model.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT)
        assert np.all(model.actuator_dyntype==mujoco.mjtDyn.mjDYN_NONE)
        assert np.all(model.actuator_gaintype==mujoco.mjtGain.mjGAIN_FIXED)
        assert np.all(model.actuator_biastype==mujoco.mjtBias.mjBIAS_AFFINE)
        assert np.all(model.actuator_forcelimited)
        np.testing.assert_array_equal(model.actuator_gear,np.tile([1.,0,0,0,0,0],(7,1)))
        joints=model.actuator_trnid[:,0]
        qids=torch.as_tensor(model.jnt_qposadr[joints].copy()).long()
        vids=torch.as_tensor(model.jnt_dofadr[joints].copy()).long()
        gains=env.action_manager.get_term("joint_stiffness");ids=gains.control_ids
        validate_arm_columns(arm_dof_cols(env),vids[ids])
        kp=env.sim.model.actuator_gainprm[:,:,0].clone()
        kd=-env.sim.model.actuator_biasprm[:,:,2].clone()
        arm_kp=apply_gains(kp,kd,ids,.5,6.4)
        limits=torch.as_tensor(model.actuator_forcerange.copy(),dtype=kp.dtype)
        original_process=gains.process_actions
        def fixed_process(values):
            original_process(values)
            gains._kp[:]=arm_kp;gains._kd[:]=6.4
        gains.process_actions=fixed_process
        hold=HeldTorqueStep(env.sim,qids,vids,kp,kd,limits,args.dt)
        env.sim.step=hold
        alive=torch.ones(env.num_envs,dtype=torch.bool)
        rows=[];actions=[];transitions=[]
        substep_original=env.metrics_manager.compute_substep
        def observe_substep():
            substep_original()
            row=sample(env)
            isolated=contact_row_qfrc(env)[:,arm_dof_cols(env)]
            assert bool(torch.isfinite(isolated).all())
            row.update(active=alive.numpy().copy(),
                contact_row_qfrc=isolated.detach().numpy().copy(),
                held_torque=hold.pd.torque.detach().numpy().copy(),
                requested_torque=hold.pd.requested.detach().numpy().copy())
            rows.append(row)
        env.metrics_manager.compute_substep=observe_substep
        first_done=np.full(env.num_envs,-1,dtype=int)
        with torch.inference_mode(),PreResetSnapshot(env) as capture:
            for step in range(25):
                assert obs["actor"].shape==(env.num_envs,42)
                raw=policy(obs)
                assert raw.shape==(env.num_envs,13) and bool(torch.isfinite(raw).all())
                actions.append(raw.detach().numpy().copy())
                before=alive.clone()
                obs,_,done,extras=wrapped.step(raw)
                outcome=extras["impact_speed_outcome"]
                assert torch.equal(outcome["done"],done.bool())
                torch.testing.assert_close(capture.latest["kp"],arm_kp.expand(env.num_envs,-1))
                torch.testing.assert_close(capture.latest["kd"],torch.full_like(capture.latest["kd"],6.4))
                record={k:v.detach().numpy().copy() for k,v in capture.latest.items()}
                record.update(active=before.numpy().copy(),done=done.detach().numpy().copy())
                for key,value in outcome.items():
                    if isinstance(value,torch.Tensor):
                        record["outcome_"+key]=value.detach().numpy().copy()
                transitions.append(record)
                ended=before & done.bool()
                first_done[ended.numpy()]=step
                alive &= ~done.bool()
                if not bool(alive.any()):break
        traces=stacked(rows)
        assert hold.count==(step+1)*cfg.decimation
        assert hold.pd.updates==(step+1)*10
        blocks=traces["held_torque"].reshape(hold.pd.updates,hold.pd.stride,env.num_envs,7)
        assert np.array_equal(blocks,np.repeat(blocks[:,:1],hold.pd.stride,axis=1))
        peak=np.where(traces["active"][:,:,None],np.abs(traces["velocity"]),0).max(axis=0)
        np.savez_compressed(args.output/"substeps.npz",**traces)
        np.savez_compressed(args.output/"transitions.npz",**stacked(transitions))
        np.savez_compressed(args.output/"actions.npz",raw=np.stack(actions),first_done=first_done)
        report=dict(task=meta["task"],dose=meta["dose"],goal=meta["goal"],physics_dt=args.dt,
            policy_dt=.02,pd_dt=.002,step_count=step+1,num_envs=env.num_envs,
            learning_updates=0,checkpoint_sha256=checkpoint_hash,original_contract=original_contract,
            intervention="Fixed gains and500Hz torque hold in separate fine-physics evaluation env",
            kp=arm_kp.tolist(),kd=[6.4]*6,reset_hashes=reset_hashes,
            runtime_identity=identity,pd_updates=hold.pd.updates,actual_torque_error=hold.max_error,
            torque_hold_bitwise=True,first_episodes_done=int((first_done>=0).sum()),
            censored_first_episodes=int((first_done<0).sum()),
            first_episode_velocity_peak=peak.tolist(),
            first_episode_speed_violations=(peak>3.1415).sum(axis=0).tolist(),
            first_episode_contact_worlds=int((traces["contact"]&traces["active"]).any(axis=0).sum()))
        (args.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    finally:
        env.close()


if __name__=="__main__":main()
