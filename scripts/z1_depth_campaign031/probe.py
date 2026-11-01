"""Qualification-only native031 fixtures and one1024-world CatPPO update.

No import-time scientific work. This entry is admitted only after canonical gates.
All fixture artifacts remain under qualification and are never training inputs.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
import copy
import hashlib
from pathlib import Path
import random
import sys
from .files import canonical,digest,need,reference,write_once

SEED=2026110199
COUNTS=dict(constructor_worlds=1036,explicit_reset_worlds=1036,vector_control_steps=120,
    world_transitions=24768,world_physics_substeps=1981440,runner_constructors=5,
    actor_forwards=52,critic_forwards=53,world_model_forwards=295968,
    algorithm_updates=1,optimizer_steps=20,save_calls=6,load_calls=8,model_constructors=20,model_constructor_observation_rows=2074,production_learn_calls=1,
    suppressed_onnx_exports=6,synthetic_owner_reset_worlds=20,
    synthetic_observer_calls=3528,synthetic_world_observations=7056)


def recipe():
    return dict(seed=SEED,device='cuda:0',tiny_worlds=2,tiny_profiles=['baseline-repeat-1','baseline-repeat-2','A0','A1','A2','A3'],
        tiny_steps_each=16,counts=dict(COUNTS),production=dict(arm='A3',worlds=1024,updates=1,policy_steps=24,
        world_transitions=24576,constructor_worlds=1024,optimizer_steps=20,actor_forwards=44,critic_forwards=45),
        overlays=['qualification seed','six2world engineering fixtures','production max_iterations=1',
                  'qualification-only ONNX export/metadata-attach suppressed; Hammer save validation retained'],
        parity=dict(method='two independently fresh baseline tapes before A0',atol_floor=1e-6,
                    repeat_multiplier=2,initial_state_rng='exact',treatment_tolerance='frozen before A0',rtol=0),
        synthetic_label='SYNTHETIC_MANAGER_FIXTURE_NOT_A_PHYSICAL_STRIKE',
        synthetic_ticks_per_arm=882,synthetic_simulated_transitions=0,
        native_autoresets_upper_bound=24768,checkpoint_reuse_in_training=False,
        production_trace=dict(depth_rows=1,population_rows=2,max_updates=1,schema_max_updates=500,additional_forwards=0))


@contextmanager
def qualification_depth_trace031(runner,output_path):
    """Capture one joined update with the unchanged500-update training schema.

    The recipe's production_trace.max_updates is the qualification capture cap;
    schema_max_updates describes each row, never a completed500-update run.
    """
    from src.tasks.hammer.rl.depth_reward031_trace import attach_credit_training_trace
    with attach_credit_training_trace(runner,output_path,max_updates=500) as trace:
        need(trace.max_updates==500 and trace.updates==0 and not trace.buffer and trace.written==0
             and trace.last_reward_call==trace.last_depth_call==0 and trace.attached and trace.stream is not None,
             'fresh qualification depth trace required')
        # Only this fresh qualification instance gets a stricter capture cap.
        trace.max_updates=1
        yield trace
        need(trace.updates==1 and not trace.buffer,'qualification depth trace requires one completed joined update')


def tensor_hash(value):
    tensor=value.detach().cpu().contiguous()
    return hashlib.sha256(str((str(tensor.dtype),tuple(tensor.shape))).encode()+tensor.numpy().tobytes()).hexdigest()


def rng_hash():
    import numpy as np
    import torch
    return dict(torch=tensor_hash(torch.random.get_rng_state()),cuda=[tensor_hash(x) for x in torch.cuda.get_rng_state_all()],
        python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),numpy=digest([np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:]]))


@contextmanager
def counters(counts):
    """Count class seams before installation so sampled-controller identity stays exact."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.hammer.mdp.sampled_pd500 import SampledPD500
    from src.tasks.hammer.mdp.depth_reward031 import NativeDepthCreditObserver031
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    import src.tasks.hammer.rl.runner as runner_module
    from src.tasks.hammer.rl.cat_ppo import CatPPO
    from rsl_rl.models import MLPModel
    originals=[];handles=[];active={'step':False};runners=[]
    def patch(obj,name,fn):
        old=getattr(obj,name);originals.append((obj,name,old,name in vars(obj)));setattr(obj,name,fn(old))
    def construct(old):
        def wrapped(env,*args,**kwargs):
            cfg=kwargs.get('cfg',args[0] if args else None)
            counts['constructor_worlds']+=cfg.scene.num_envs
            need(counts['constructor_worlds']<=COUNTS['constructor_worlds'],'constructor budget exceeded')
            old(env,*args,**kwargs)
        return wrapped
    patch(ManagerBasedRlEnv,'__init__',construct)
    def step(old):
        def wrapped(env,*args,**kwargs):
            counts['vector_control_steps']+=1;counts['world_transitions']+=env.num_envs
            need(counts['world_transitions']<=COUNTS['world_transitions'],'transition budget exceeded')
            active['step']=True
            try:return old(env,*args,**kwargs)
            finally:active['step']=False
        return wrapped
    patch(ManagerBasedRlEnv,'step',step)
    def reset(old):
        def wrapped(controller,*args,**kwargs):
            counts['explicit_reset_worlds']+=controller.env.num_envs
            return old(controller,*args,**kwargs)
        return wrapped
    patch(SampledPD500,'reset_env',reset)
    def native(old):
        def wrapped(controller,ids=None,*args,**kwargs):
            if active['step']:counts['native_autoreset_worlds']+=controller.env.num_envs if ids is None else len(ids)
            return old(controller,ids,*args,**kwargs)
        return wrapped
    patch(SampledPD500,'reset_idx',native)
    def physics(old):
        def wrapped(controller,*args,**kwargs):
            counts['world_physics_substeps']+=controller.env.num_envs
            return old(controller,*args,**kwargs)
        return wrapped
    patch(SampledPD500,'step',physics)
    def observer(old):
        def wrapped(owner,*args,**kwargs):
            if counts['synthetic_active']:
                counts['synthetic_observer_calls']+=1;counts['synthetic_world_observations']+=owner.num_envs
            else:counts['physical_depth_observer_calls']+=1
            return old(owner,*args,**kwargs)
        return wrapped
    patch(NativeDepthCreditObserver031,'__call__',observer)
    def runner_init(old):
        def wrapped(runner,*args,**kwargs):
            old(runner,*args,**kwargs);counts['runner_constructors']+=1;runners.append(runner)
            need(type(runner.alg) is CatPPO,'actual CatPPO required')
            for name in ('actor','critic'):
                def called(model,inputs,kind=name):
                    counts[kind+'_forwards']+=1
                    batch=next(iter(inputs[0].values())).shape[0]
                    counts['world_model_forwards']+=batch
                    need(counts[kind+'_forwards']<=COUNTS[kind+'_forwards'],'forward budget exceeded')
                handles.append(getattr(runner.alg,name).register_forward_pre_hook(called))
            original_step=runner.alg.optimizer.step
            def optimizer_step(*a,**kw):
                counts['optimizer_steps']+=1
                need(counts['optimizer_steps']<=20,'optimizer step budget exceeded')
                return original_step(*a,**kw)
            runner.alg.optimizer.step=optimizer_step
        return wrapped
    patch(HammerOnPolicyRunner,'__init__',runner_init)
    def model_init(old):
        def wrapped(model,obs,*args,**kwargs):
            counts['model_constructors']+=1
            counts['model_constructor_observation_rows']+=next(iter(obs.values())).shape[0]
            return old(model,obs,*args,**kwargs)
        return wrapped
    patch(MLPModel,'__init__',model_init)
    def learn(old):
        def wrapped(runner,num_learning_iterations,init_at_random_ep_len=False):
            from scripts.z1_exploration_pilot_training import _initialize
            from .manifest import training_cell
            counts['production_learn_calls']+=1
            need(num_learning_iterations==1 and runner.env.num_envs==1024 and runner.env.unwrapped.common_step_counter==0
                 and not runner.alg.optimizer.state,'fresh production qualification only')
            cell=training_cell(3);cell['seed']=SEED
            runner._qualification_initial031=_initialize(runner,cell)
            from src.tasks.hammer.rl.depth_reward031_trace import validate_depth_trace_row031
            from src.tasks.hammer.rl.velocity_population_trace import attach_velocity_population_trace
            trace_root=Path(runner.logger.log_dir).parent
            with qualification_depth_trace031(runner,trace_root/'qualification-depth-trace.jsonl'), \
                 attach_velocity_population_trace(runner,trace_root/'qualification-population.jsonl',max_updates=1):
                result=old(runner,num_learning_iterations,init_at_random_ep_len)
            import json
            depth_rows=[json.loads(line) for line in (trace_root/'qualification-depth-trace.jsonl').read_text().splitlines()]
            population_rows=[json.loads(line) for line in (trace_root/'qualification-population.jsonl').read_text().splitlines()]
            need(len(depth_rows)==1 and len(population_rows)==2,'qualification trace coverage differs')
            validate_depth_trace_row031(depth_rows[0],0)
            need(runner.env.unwrapped.common_step_counter==24 and runner.current_learning_iteration==0
                 and runner.alg.storage.step==0 and runner.alg.optimizer.state,'production PPO completion differs')
            for model in (runner.alg.actor,runner.alg.critic):
                need(all(bool(v.isfinite().all()) for v in model.state_dict().values()),'nonfinite production state')
            return result
        return wrapped
    patch(HammerOnPolicyRunner,'learn',learn)
    def update(old):
        def wrapped(alg,*args,**kwargs):
            counts['algorithm_updates']+=1;need(counts['algorithm_updates']==1,'one PPO update only')
            return old(alg,*args,**kwargs)
        return wrapped
    patch(CatPPO,'update',update)
    for method,key in (('save','save_calls'),('load','load_calls')):
        def count_method(old,key=key):
            def wrapped(self,*args,**kwargs):
                counts[key]+=1;result=old(self,*args,**kwargs)
                if key=='save_calls':counts.setdefault('actual_save_receipts',[]).append(dict(file=reference(args[0]),counter=self.env.unwrapped.common_step_counter,iteration=self.current_learning_iteration))
                return result
            return wrapped
        patch(HammerOnPolicyRunner,method,count_method)
    def no_export(old):
        def wrapped(*args,**kwargs):counts['suppressed_onnx_exports']+=1
        return wrapped
    patch(HammerOnPolicyRunner,'export_policy_to_onnx',no_export)
    patch(runner_module,'attach_metadata_to_onnx',lambda old:lambda *args,**kwargs:None)
    try:yield runners
    finally:
        for handle in handles:handle.remove()
        for obj,name,old,existed in reversed(originals):
            if existed:setattr(obj,name,old)
            else:delattr(obj,name)
        for runner in runners:
            # No later use of qualification optimizers; release model references.
            if not getattr(runner.env,'_qualification_closed031',False):
                runner.env.close();runner.env._qualification_closed031=True


def make_tiny(arm):
    import torch
    from scripts.z1_depth_reward_training031 import build_train_config
    from scripts.z1_split_credit_training021 import build_train_config as baseline
    from scripts.train import configure_torch_backends,configure_impact_speed_experiment,configure_impact_speed_controller,install_impact_speed_controller,wrap_impact_speed_env,ManagerBasedRlEnv
    task,cfg=baseline(0) if arm=='baseline' else build_train_config(int(arm[1]))
    need(cfg.env.scene.num_envs==1024 and cfg.agent.max_iterations==500 and cfg.agent.num_steps_per_env==24 and cfg.agent.resume is False,'original production factory required')
    cfg.env.scene.num_envs=2;cfg.env.seed=cfg.agent.seed=SEED;cfg.env.sim.nan_guard.output_dir=str(Path.cwd()/'nan-dumps')
    configure_torch_backends()
    configure_impact_speed_experiment(cfg.env,bounds=cfg.speed_bounds,cost=cfg.speed_cost,mode=cfg.speed_mode)
    configure_impact_speed_controller(cfg.env,sampled_pd500=True,gain_mode='variable')
    raw=ManagerBasedRlEnv(cfg.env,device='cuda:0')
    install_impact_speed_controller(raw,sampled_pd500=True,gain_mode='variable')
    vec=wrap_impact_speed_env(raw,clip_actions=cfg.agent.clip_actions)
    return task,cfg,raw,vec


def initial_snapshot(raw):
    return dict(qpos=tensor_hash(raw.sim.data.qpos),qvel=tensor_hash(raw.sim.data.qvel),
        observations={k:tensor_hash(v) for k,v in raw.obs_buf.items()},rng=rng_hash(),
        common_step_counter=raw.common_step_counter,sim_step_counter=raw._sim_step_counter)


def native_tape(raw,vec):
    import torch
    from src.tasks.hammer.cat.keys import CAT_DELTA_KEY,CAT_R_POS_KEY
    rows=[]
    for _ in range(16):
        action=torch.zeros((2,13),device='cuda:0')
        obs,reward,done,extras=vec.step(action)
        row=dict(qpos=raw.sim.data.qpos.detach().cpu().clone(),qvel=raw.sim.data.qvel.detach().cpu().clone(),
            reward=reward.detach().cpu().clone(),done=done.detach().cpu().clone(),action=action.cpu(),
            kp=raw._sampled_pd500.applied_kp.detach().cpu().clone(),kd=raw._sampled_pd500.applied_kd.detach().cpu().clone())
        for key in ('applied_joint_position','requested_joint_position','torque','requested_torque','saturated','physics_substeps_since_reset'):
            row['controller/'+key]=getattr(raw._sampled_pd500,key).detach().cpu().clone()
        row['native_action_buffer']=raw.action_manager.action.detach().cpu().clone()
        for key,value in raw.obs_buf.items():row['observations/'+key]=value.detach().cpu().clone()
        for key,extras_key in (('delta',CAT_DELTA_KEY),('r_pos',CAT_R_POS_KEY)):
            value=extras.get(extras_key)
            need(value is not None,'required native parity field missing: '+key)
            row[key]=value.detach().cpu().clone()
        for key in ('reset_terminated','reset_time_outs'):
            value=getattr(raw,key,None)
            need(value is not None,'required native parity field missing: '+key)
            row[key]=value.detach().cpu().clone()
        row['rng_hash']=rng_hash()
        for j,name in enumerate(raw.reward_manager.active_terms):
            if not name.startswith('native_depth_'):row['reward/'+name]=raw.reward_manager._step_reward[:,j].detach().cpu().clone()
        rows.append(row)
    return rows


def tolerance(first,second):
    import torch
    result={}
    for key in first[0]:
        if key=='rng_hash':
            need(all(a[key]==b[key] for a,b in zip(first,second)),'per-step baseline RNG differs');continue
        delta=max(float((a[key].double()-b[key].double()).abs().max()) for a,b in zip(first,second))
        exact=key in ('action','kp','kd','done','native_action_buffer','reset_terminated','reset_time_outs','controller/applied_joint_position','controller/requested_joint_position','controller/saturated')
        if exact:need(delta==0,'baseline exact target/gain/mask repeat differs')
        result[key]=0. if exact else max(1e-6,2*delta)
    return result


def compare_tape(first,actual,bounds):
    maxima={}
    need(all(a['rng_hash']==b['rng_hash'] for a,b in zip(first,actual)),'per-step A0 RNG differs')
    for key,bound in bounds.items():
        delta=max(float((a[key].double()-b[key].double()).abs().max()) for a,b in zip(first,actual))
        need(delta<=bound,'A0 passive parity failed: '+key);maxima[key]=delta
    return maxima


def reload_fixture(arm,cfg,vec,output):
    import torch
    from mjlab.tasks.registry import load_runner_cls
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata,validate_impact_speed_metadata
    runner=load_runner_cls('Unitree-Z1-Hammer-Spline-DepthReward031-'+arm)(vec,asdict(cfg.agent),None,'cuda:0')
    from scripts.z1_exploration_pilot_training import _initialize
    from .manifest import training_cell
    cell=training_cell(int(arm[1]));cell['seed']=SEED
    initialized=_initialize(runner,cell)
    runner.logger.init_logging_writer()
    runner.alg.eval_mode();obs=vec.get_observations()
    before=(runner.alg.actor(obs,stochastic_output=False).detach().clone(),runner.alg.critic(obs).detach().clone())
    model_hashes={name:{k:tensor_hash(v) for k,v in getattr(runner.alg,name).state_dict().items()} for name in ('actor','critic')}
    path=output/('qualification-only-'+arm+'.pt');runner.save(str(path),infos={'qualification_only031':True})
    payload=torch.load(path,map_location='cuda:0',weights_only=False)
    wrong_payload=copy.deepcopy(payload);wrong_payload['infos']['impact_speed_contract']['spline_depth_reward031']['arm']='A3' if arm!='A3' else 'A0'
    wrong_path=output/('qualification-only-wrong-arm-'+arm+'.pt');torch.save(wrong_payload,wrong_path)
    before_negative={name:{k:tensor_hash(v) for k,v in getattr(runner.alg,name).state_dict().items()} for name in ('actor','critic')}
    before_counter=vec.unwrapped.common_step_counter
    try:runner.load(str(wrong_path),load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False,'rnd':False},strict=True,map_location='cuda:0')
    except ValueError:pass
    else:raise ValueError('wrong-arm checkpoint loader admitted')
    need(before_counter==vec.unwrapped.common_step_counter and before_negative=={name:{k:tensor_hash(v) for k,v in getattr(runner.alg,name).state_dict().items()} for name in ('actor','critic')},'negative load mutated state')
    with torch.no_grad():
        next(runner.alg.actor.parameters()).add_(.25);next(runner.alg.critic.parameters()).add_(.25)
    runner.load(str(path),load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False,'rnd':False},strict=True,map_location='cuda:0')
    after=(runner.alg.actor(obs,stochastic_output=False).detach().clone(),runner.alg.critic(obs).detach().clone())
    need(all(torch.equal(a,b) for a,b in zip(before,after)),'deterministic reload output differs')
    need(model_hashes=={name:{k:tensor_hash(v) for k,v in getattr(runner.alg,name).state_dict().items()} for name in ('actor','critic')},'reload state tensors differ')
    saved=build_impact_speed_metadata(vec.unwrapped,vec.clip_actions);wrong=copy.deepcopy(saved)
    wrong['spline_depth_reward031']['arm']='A3' if arm!='A3' else 'A0'
    try:validate_impact_speed_metadata(wrong,saved)
    except ValueError:pass
    else:raise ValueError('wrong-arm metadata admitted')
    return dict(arm=arm,checkpoint=reference(path),model_hashes=model_hashes,metadata=saved,wrong_arm_rejected=True,
        initialized=initialized,negative_checkpoint=reference(wrong_path),negative_load_state_unchanged=True)


def synthetic_fixture(raw,arm,counts):
    import torch
    from mjlab.managers.reward_manager import RewardManager
    from src.tasks.hammer.mdp.depth_reward031 import depth_collect_metric031,depth_progress_reward031,depth_completion_reward031
    owner=raw._z1_depth_credit031;tracker=raw._hammer_first_strike
    manager=RewardManager({k:v for k,v in raw.cfg.rewards.items() if k.startswith('native_depth_')},raw)
    depth=raw.scene['nail_block'].data.joint_pos
    def reset(ids=None):
        owner.reset(ids);counts['synthetic_owner_reset_worlds']+=2 if ids is None else len(ids)
    def observe(values):
        depth[:,owner.nail_cfg.joint_ids]=torch.tensor(values,device=raw.device).reshape(2,1)
        owner(raw,arm=arm,nail_cfg=owner.nail_cfg)
    def boundary():
        raw.common_step_counter+=1;paid=manager.compute(dt=raw.step_dt);depth_collect_metric031(raw);return paid
    counts['synthetic_active']=True
    try:
        depth[:,owner.nail_cfg.joint_ids]=0;reset();tracker.started[:]=True;tracker.first_contact_time_s[:]=.00025
        tracker._substep_count[:]=0;raw.reset_buf[:]=False;total=torch.zeros(2,device=raw.device)
        for tick in range(1,881):
            tracker._substep_count[:]=tick;observe([.030 if tick==281 else .001,.017])
            if tick%80==0:
                raw.reset_buf[:]=tick==880;total+=boundary()
        expected={'A0':[0.,0.],'A1':[1.,.5],'A2':[1.,0.],'A3':[2.,.5]}[arm]
        torch.testing.assert_close(total,torch.tensor(expected,device=raw.device),rtol=0,atol=1e-6)
        snap=owner.last_snapshot
        need(snap['first_attainment_tick'].tolist()==[281,-1] and snap['physics_sample_count'].tolist()==[880,880],'synthetic transient history differs')
        for func in (depth_progress_reward031,depth_completion_reward031):func(raw);func(raw)
        need(torch.equal(snap['completion_paid'],owner.last_snapshot['completion_paid']),'idempotent readers differ')
        depth[:,owner.nail_cfg.joint_ids]=0;reset();observe([.030,.030]);tracker._substep_count[:]=torch.tensor([721,801],device=raw.device)
        raw.reset_buf[:]=True;boundary();timeout=owner.last_snapshot
        completion=arm in ('A2','A3')
        need(timeout['completion_paid'].tolist()==[0.,float(completion)] and timeout['completion_forfeited'].tolist()==[float(completion),0.], 'timeout/horizon precedence differs')
        old_generation=timeout['generation'].clone();depth[0,owner.nail_cfg.joint_ids]=.017;reset(torch.tensor([0],device=raw.device))
        need(torch.equal(owner.last_snapshot['generation'],old_generation),'pre-reset snapshot not preserved')
        tracker._substep_count[:]=80;raw.reset_buf[:]=False;observe([.030,.030]);boundary();selected=owner.last_snapshot
        need((selected['generation']-old_generation).tolist()==[1,0],'subset reset changed another world')
        return dict(label='SYNTHETIC_MANAGER_FIXTURE_NOT_A_PHYSICAL_STRIKE',arm=arm,total=total.tolist(),
            transient={k:v.tolist() for k,v in snap.items()},timeout={k:v.tolist() for k,v in timeout.items()},
            selected_reset={k:v.tolist() for k,v in selected.items()},simulated_transitions=0)
    finally:counts['synthetic_active']=False


def execute_probe(output,qualification_recipe,*,before_science=None):
    import torch
    from scripts.z1_depth_reward_training031 import build_train_config
    from scripts.train import run_train
    from scripts import z1_split_credit_training021  # Baseline setup must precede the origin check.
    need(torch.cuda.is_available(),'CUDA probe requires actual GPU')
    expected=qualification_recipe['gates'][3]['probe']
    need(expected==recipe(),'reviewed exact probe budget differs')
    output=Path(output);counts={key:0 for key in COUNTS};counts.update(native_autoreset_worlds=0,physical_depth_observer_calls=0,synthetic_active=False)
    results=[];baselines=[];initial=None;bounds=None;active=[]
    try:
        with counters(counts) as runners:
            if before_science is not None:before_science()
            for name in recipe()['tiny_profiles']:
                arm='baseline' if name.startswith('baseline') else name
                task,cfg,raw,vec=make_tiny(arm);active.append(vec)
                actual_initial=initial_snapshot(raw)
                if initial is None:initial=actual_initial
                else:need(actual_initial==initial,'fresh baseline/A0–A3 initial physical/RNG state differs')
                tape=native_tape(raw,vec);rng=rng_hash();torch.save(tape,output/(name+'-native-tape.pt'))
                if arm=='baseline':
                    baselines.append((tape,rng))
                    if len(baselines)==2:
                        need(baselines[0][1]==rng,'baseline repeat RNG differs')
                        bounds=tolerance(baselines[0][0],tape);write_once(output/'frozen-parity-tolerances.json',bounds)
                    results.append(dict(profile=name,initial=actual_initial,final_rng=rng))
                else:
                    row=dict(profile=name,initial=actual_initial,final_rng=rng)
                    if arm=='A0':
                        need(rng==baselines[0][1],'A0 observer consumed RNG');row['parity_maxima']=compare_tape(baselines[0][0],tape,bounds)
                    row['reload']=reload_fixture(arm,cfg,vec,output)
                    row['synthetic']=synthetic_fixture(raw,arm,counts);results.append(row)
                vec.close();vec._qualification_closed031=True;active.remove(vec)
            task,cfg=build_train_config(3)
            need(cfg.env.scene.num_envs==1024 and cfg.agent.max_iterations==500 and cfg.agent.num_steps_per_env==24 and cfg.agent.resume is False,'production factory changed')
            cfg.env.seed=cfg.agent.seed=SEED;cfg.agent.max_iterations=1;cfg.env.sim.nan_guard.output_dir=str(Path.cwd()/'nan-dumps')
            run_train(task,cfg,output/'production1024-qualification-only')
            production_initial=runners[-1]._qualification_initial031
            initial_models=[row['reload']['model_hashes'] for row in results if 'reload' in row]
            need(all(row==initial_models[0] for row in initial_models),'matched model/normalizer initialization differs across031 arms')
            for key,value in COUNTS.items():need(counts[key]==value,'actual qualification count differs: '+key+f' {counts[key]} != {value}')
            need(counts['physical_depth_observer_calls']==7040,'actual4kHz depth observer callback count differs')
            need(counts['native_autoreset_worlds']<=24768,'native reset bound exceeded')
        result=dict(status='PROBE_ASSERTIONS_PASSED',qualification_only=True,counts=counts,profiles=results,
            production_checkpoint=reference(output/'production1024-qualification-only/model_0.pt'),
            production_initial=production_initial,
            qualification_depth_trace=reference(output/'qualification-depth-trace.jsonl'),
            qualification_population=reference(output/'qualification-population.jsonl'),
            cuda_device=torch.cuda.get_device_name(0),recipe_sha256=digest(recipe()),checkpoint_reused_for_training=False)
        write_once(output/'probe-result.json',result)
        return result
    finally:
        for env in active:env.close()
        write_once(output/'probe-actual-counts.json',counts)
