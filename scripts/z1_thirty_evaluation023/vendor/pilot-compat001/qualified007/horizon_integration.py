"""Current evaluator integration candidate; never invoked by pure checks.

No task aliases, checkpoint metadata rewrites, or old-source evaluator imports.
The admitted source root must contain the matching current training task.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
CASES=(("original-route-1",-1,-.02),("start-minus10mm",0,-.01),
       ("original-route+0",0,0.),("start-plus10mm",0,.01),("original-route+1",1,.02))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def durable(path,value):
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())


def tree_bytes(root):
    total=0
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('output symlink forbidden')
        if path.is_file():total+=path.stat().st_size
    return total


def persist_observer_mapping(output,mapping):
    """Seal mapping/provenance before physics; return portable row binding."""
    path=Path(output)/'observer-mapping.json'
    durable(path,mapping)
    return {'path':str(path.resolve()),'sha256':sha(path)}


class HorizonAdapter:
    def __init__(self,intent,case,output):
        self.intent=intent;self.case=case;self.output=output;self.stack=ExitStack()
        self.observer=None;self.overlay=None;self.first_step=False;self.closed=False
        self.pending_reset_worlds=set();self.started_counts={};self.reset_event_count=0

    def bind(self,env,runner,saved,contract):
        # Evaluator called the real strict validator before reaching this seam.
        from src.tasks.hammer.rl.impact_speed_contract import validate_impact_speed_metadata
        validate_impact_speed_metadata(saved,contract)
        from src.tasks.hammer.mdp.spline_horizon import horizon_window_complete
        term=env.termination_manager.get_term_cfg('nail_driven')
        window=self.intent['policy']['native_window_s']
        if term.func is not horizon_window_complete or term.params!={'window_s':window}:
            raise ValueError('integer native task predicate/window differs before overlay')
        if env.cfg.episode_length_s!=20. or env.physics_dt!=.00025:
            raise ValueError('unchanged finite training horizon/physics required')
        self.env=env;self.runner=runner;self.contract=contract
        self.actor_before=self.tensor_hash(runner.alg.actor.state_dict())
        self.normalizer_before=self.tensor_hash(runner.alg.actor.obs_normalizer.state_dict())
        self.native_func=term.func

    @staticmethod
    def tensor_hash(state):
        digest=hashlib.sha256()
        for name,value in sorted(state.items()):
            digest.update(name.encode());digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def __enter__(self):
        env=self.env;cfg=env.event_manager.get_term_cfg('sample_strike_route_signs')
        self.route_cfg=cfg;self.original_route=cfg.func
        def forced_route(source,env_ids,**params):
            result=self.original_route(source,env_ids,**params)
            source._strike_reference._route_sign[env_ids]=self.case[1]
            worlds=env_ids.detach().cpu().tolist()
            self.pending_reset_worlds.update(worlds)
            durable(self.output/f'reset-event-{self.reset_event_count:05d}.json',
                {'worlds':worlds,'phase':'native route reset event; full reset completion not yet established'})
            self.reset_event_count+=1
            return result
        self.route_wrapper=forced_route;cfg.func=forced_route
        self.stack.callback(setattr,cfg,'func',self.original_route)
        try:
            if self.case[0].startswith('start-'):
                from reset_pose import ResetPoseContext
                pose=self.intent['pose_catalog'][self.case[0]]
                if pose['offset_m']!=self.case[2]:raise ValueError('intermediate pose offset differs')
                term=env.event_manager.get_term_cfg('reset_strike_route_joints')
                self.stack.enter_context(ResetPoseContext(env,pose['executed_joint_positions_rad'],term.params['route_joint_positions']))
        except BaseException:
            self.stack.close()
            raise
        return self

    def after_reset(self,obs):
        import torch
        from safety_observer import SafetyAccumulator
        if not bool((self.env._strike_reference.route_signs()==self.case[1]).all()):
            raise ValueError('realized route differs')
        geometry=None
        if self.case[0].startswith('start-'):
            from reset_pose import verify_reset_geometry
            geometry=verify_reset_geometry(self.env,self.intent['pose_catalog'][self.case[0]])
        # Durable lifecycle before action and observer installation. Every world
        # is classified reset_observed only when this actual callback completed.
        durable(self.output/'reset-observed.json',{'case_id':self.case[0],
            'worlds':list(range(self.env.num_envs)),'geometry':geometry,
            'observation_sha256':self.tensor_hash(dict(obs)),
            'actor_sha256':self.actor_before,'normalizer_sha256':self.normalizer_before})
        self.observer=SafetyAccumulator(self.env.num_envs,self.env.device,
            self.env.physics_dt,native_window_s=self.intent['policy']['native_window_s'])
        self.stack.enter_context(self.observer.install(self.env))
        self.mapping_binding=persist_observer_mapping(self.output,self.observer.mapping_record)

    def protocol(self,original):
        # Using the native protocol is essential: its float 50ms common-window
        # success suppression must not replace the task's integer predicate.
        from continuation_tools import NativeBoundaryContinuation
        from src.tasks.hammer.mdp.strike_evaluation import evaluation_snapshot
        if self.env.cfg.auto_reset is not True:
            raise ValueError('training auto_reset contract must be True before measurement overlay')
        self.original_auto_reset=self.env.cfg.auto_reset
        self.env.cfg.auto_reset=False
        self.stack.callback(setattr,self.env.cfg,'auto_reset',self.original_auto_reset)
        if hasattr(self.env,'_manual_reset_pending'):
            self.stack.callback(self.env._manual_reset_pending.copy_,self.env._manual_reset_pending.clone())
        self.overlay=NativeBoundaryContinuation(self.env,accumulator=self.observer,
            native_snapshot=evaluation_snapshot,native_window_s=self.intent['policy']['native_window_s'])
        @contextmanager
        def combined():
            # Native MeasurementProtocol runs the original integer predicate
            # once and records its evidence; it returns that same native mask.
            with original, self.overlay:
                yield self.overlay
        return combined()

    def before_step(self,step):
        if time.time()>=self.intent['deadline_epoch']:raise TimeoutError('finite worker deadline')
        if tree_bytes(Path(self.intent['output_root']))>self.intent['max_output_bytes']:
            raise RuntimeError('finite output cap')
        # mjlab versions with a manual-reset guard otherwise reject continuing
        # vector physics after an observed first completion. Clear ONLY the
        # runtime pending bit for observer-closed worlds; call no reset method.
        pending=getattr(getattr(self,'env',None),'_manual_reset_pending',None)
        if pending is not None:
            if bool((pending & self.observer.active).any()):
                raise ValueError('manual reset pending on still-observed first attempt')
            pending.zero_()
        self.step_starts=sorted(self.pending_reset_worlds)
        if any(self.started_counts.get(world,0)>0 for world in self.step_starts):
            raise RuntimeError('held before any second-episode physical start; inert-world handling required')
        self.pending_reset_worlds.clear()

    def after_step(self,step,done):
        if self.step_starts:
            for world in self.step_starts:self.started_counts[world]=self.started_counts.get(world,0)+1
            durable(self.output/f'physical-start-step-{step:05d}.json',
                {'worlds':self.step_starts,'counts':self.started_counts,'policy_step':step})
            if any(value>1 for value in self.started_counts.values()):
                raise RuntimeError('additional physical first start observed: candidate requires reviewed inert-world handling')
        if not self.first_step:
            durable(self.output/'first-step-observed.json',{'case_id':self.case[0],
                'worlds':list(range(self.env.num_envs)),'policy_step':step,
                'observed_substeps':self.observer.observed.cpu().tolist()})
            self.first_step=True
        # Compact partial state overwritten atomically; lifecycle transitions
        # above are immutable. Arrays are bounded by 32 worlds x six channels.
        temp=self.output/'partial.tmp'
        temp.write_text(json.dumps({'step':step,'observed_samples':self.observer.observed.cpu().tolist(),
            'active':self.observer.active.cpu().tolist(),
            'accepted_contact_sample':self.observer.onset.cpu().tolist(),
            'count_including_onset':self.observer.count.cpu().tolist(),
            'end_reason':list(self.observer.end_reasons)},allow_nan=False))
        os.replace(temp,self.output/'partial.json')

    def __exit__(self,*exc):
        if self.closed:return False
        self.closed=True
        try:
            if self.observer is not None and hasattr(self,'mapping_binding'):
                if sha(self.mapping_binding['path'])!=self.mapping_binding['sha256']:
                    raise ValueError('observer mapping/provenance changed during evaluation')
                records=self.observer.records()
                self.trace_binding=self.observer.onset_trace.save(self.output/'onset-native-producers.npz')
                from continuation_tools import VELOCITY_LIMIT,LOAD_CAPS
                endpoints=self.observer.onset_trace.safety_summaries(velocity_limit=VELOCITY_LIMIT,load_caps=LOAD_CAPS)
                endpoints['native_onset_trace']=self.trace_binding
                endpoints['observer_mapping']=self.mapping_binding
                endpoint_path=self.output/'inclusive-safety-endpoints.json'
                durable(endpoint_path,endpoints)
                self.endpoint_binding={'path':str(endpoint_path.resolve()),'sha256':sha(endpoint_path)}
                for row in records:
                    row['observer_mapping']=self.mapping_binding
                    row['native_onset_trace']=self.trace_binding
                    row['safety_endpoint_summary']=self.endpoint_binding
                    row['legacy_window_intervals']={'contact_0_50ms':'[0,50ms)',
                        'contact_50_200ms':'[50ms,200ms)','contact_0_200ms':'[0,200ms)'}
                    row['authoritative_contact_0_200ms_inclusive']=endpoints['worlds'][row['world_id']]['windows']['authoritative_0_200ms_inclusive']
                durable(self.output/'observer-records.json',records)
                if self.overlay is not None:
                    native_records=self.overlay.records()
                    for row in native_records:
                        row['observer_mapping']=self.mapping_binding
                        row['native_onset_trace']=self.trace_binding
                        row['safety_endpoint_summary']=self.endpoint_binding
                    durable(self.output/'native-boundary-records.json',native_records)
        finally:
            self.stack.close()
        if self.route_cfg.func is not self.original_route or self.env.termination_manager.get_term_cfg('nail_driven').func is not self.native_func:
            raise ValueError('callbacks not restored')
        if self.tensor_hash(self.runner.alg.actor.state_dict())!=self.actor_before or self.tensor_hash(self.runner.alg.actor.obs_normalizer.state_dict())!=self.normalizer_before:
            raise ValueError('actor/normalizer mutated')
        return False


def execute_horizon_intent(intent):
    source=Path(intent['source_root']).resolve(strict=True)
    required={'horizon_integration.py','current_evaluator_candidate.py','current-evaluator-origin.json',
              'reset_pose.py','safety_observer.py','continuation_tools.py','onset_trace.py'}
    packages={Path(row['path']).resolve():row['sha256'] for row in intent['package_files']}
    if any((HERE/name).resolve() not in packages or sha(HERE/name)!=packages[(HERE/name).resolve()] for name in required):
        raise ValueError('complete local integration closure must be pinned')
    for name in sys.modules:
        if name=='src' or name.startswith('src.') or name=='scripts' or name.startswith('scripts.'):
            raise ValueError('ambient training/evaluator modules loaded before frozen imports')
    expected={Path(row['path']).resolve():row['sha256'] for row in intent['source_files']}
    actual={p.resolve() for folder in ('src','scripts') for p in (source/folder).rglob('*.py')}
    if not actual.issubset(expected) or any(sha(p)!=expected[p] for p in actual):
        raise ValueError('complete current Python source inventory required')
    if sha(source/'scripts/evaluate_impact_speed.py')!=json.loads((HERE/'current-evaluator-origin.json').read_text())['sha256']:
        raise ValueError('current evaluator base differs from reviewed adapter patch')
    sys.path[:0]=[str(HERE),str(source)]
    import torch
    runtime={'python':sys.version,'executable':sys.executable,'platform':sys.platform,
        'packages':{name:importlib.metadata.version(name) for name in ('torch','mujoco','mujoco-warp','mjlab')},
        'torch_cuda':torch.version.cuda,'cuda_device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if runtime!=intent['runtime_identity']:raise ValueError('actual CPU runtime differs from sealed identity')
    durable(Path(intent['output_root'])/'actual-runtime.json',runtime)
    import mjlab.tasks
    import src.tasks
    if not Path(sys.modules['src.tasks'].__file__).resolve().is_relative_to(source):
        raise ValueError('task import origin differs')
    spec=importlib.util.spec_from_file_location('horizon_current_evaluator',HERE/'current_evaluator_candidate.py')
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    chosen=[row for row in CASES if row[0] in intent['cases']]
    if [row[0] for row in chosen]!=intent['cases'] or len(chosen)!=(1 if intent['scope']=='own_cpu32' else 5):
        raise ValueError('declared fixed cases/order differ')
    root=Path(intent['output_root']);reports=[]
    for case in chosen:
        if sha(intent['policy']['checkpoint_path'])!=intent['policy']['checkpoint_sha256']:
            raise ValueError('checkpoint changed before physical-start reload')
        out=root/case[0]
        args=evaluator.parse_args(['--task',intent['policy']['task'],'--checkpoint',intent['policy']['checkpoint_path'],
            '--speed-bounds','1.0','1.5','--speed-cost','0.8493676','--seed',str(intent['evaluation_seed']),
            '--num-envs','32','--mode','policy','--output',str(out),'--device','cpu',
            '--sampled-pd500','--sampled-pd500-gain-mode','variable','--measurement-protocol','native'])
        out.mkdir(exist_ok=False)
        args.horizon_adapter=HorizonAdapter(intent,case,out)
        rows,report=evaluator._condition(args,None,0)
        if args.horizon_adapter.started_counts!={world:1 for world in range(32)}:
            raise ValueError('actual physical first-start population differs')
        mapping=args.horizon_adapter.mapping_binding
        trace=args.horizon_adapter.trace_binding
        endpoints=args.horizon_adapter.endpoint_binding
        for row in rows:
            row['observer_mapping']=mapping
            row['native_onset_trace']=trace
            row['safety_endpoint_summary']=endpoints
        report['observer_mapping']=mapping
        report['native_onset_trace']=trace
        report['safety_endpoint_summary']=endpoints
        durable(out/'episodes.json',rows);durable(out/'report.json',report)
        reports.append({'case_id':case[0],'report_sha256':sha(out/'report.json'),'observer_mapping':mapping,'native_onset_trace':trace,'safety_endpoint_summary':endpoints})
    if sha(intent['policy']['checkpoint_path'])!=intent['policy']['checkpoint_sha256']:
        raise ValueError('checkpoint changed during evaluation')
    return {'status':'RAW_REVIEW_REQUIRED','attempts':len(chosen)*32,'reports':reports,
        'observer_mappings':[report['observer_mapping'] for report in reports],
        'native_onset_traces':[report['native_onset_trace'] for report in reports],
        'safety_endpoint_summaries':[report['safety_endpoint_summary'] for report in reports],
        'checkpoint_contract_validated_before_overlay':True,'final_contract_restored':True,
        'reset_frames_sampled':0,'scientific_qualification':False}
