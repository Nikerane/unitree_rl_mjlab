"""Opt-in eight-screen envelope around byte-identical qualified CPU32 measurements.

Importing this file does not import torch/tasks or run evaluation. The single
execute entry requires root's admitted bounded intent; authority/claim/process
ownership remains the caller's finite supervisor, not this module.
"""
from __future__ import annotations
import importlib.metadata
import ast
import json
import os
from pathlib import Path
import sys
import time
import types

from pilot_contract import canonical,need,sha,validate_record,validate_payload,evaluator_arguments

HERE=Path(__file__).resolve().parent
QUALIFIED_PINS={
 'horizon_worker.py':'3f15e8d56e2c52d916c1f2195460fc6e59bbd42840738be37acf67358be79ff9',
 'horizon_integration.py':'ed9d28c7781ed5bb776049d9cca52536f51c405215bd0c3d18395defc35ee3d7',
 'supervisor/cpu_supervisor.py':'b6a2d726bb58ed63451567089ec5e9b295bed3b979f427091d42becb5c03c757',
 'current_evaluator_candidate.py':'1002720dbf69427b0173acd6b90eb23b58909b19f955aca37d520e201ba9a2e3',
 'continuation_tools.py':'0b6484499258ee0e1b03feae81a533b0bfad35d0d2810dc8e78031a858f7c902',
 'onset_trace.py':'16137ae895ec33f9898b4621b067eb6b10f665d997d14fabf062fc42488b147e',
 'safety_observer.py':'b6c361499315bb5ccbf7780c2a9c52802e35e9fd27853694d8651137340e4f95',
 'reset_pose.py':'1d10a5f7abd1efb63ff736025d8414a240d953be68ede3cbac6651d3b5b25d81',
 'current-evaluator-origin.json':'482b4259654b822db5d3f54863d742494d14f29398c45fa2e6d338d19d2a86d9',
 'horizon_eval_preflight.py':'2fdd315f2978e7079d33a40421a89b038918b166720b457a2eef2548fe60bd3d',
}


def durable(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def reference(ref):
    need(type(ref)is dict and set(ref)=={'path','sha256'},'exact hashed file reference required')
    p=Path(ref['path']);need(p.is_absolute() and not p.is_symlink() and p.is_file(),'absolute regular bound file required')
    need(sha(p)==ref['sha256'],'bound file SHA differs: '+str(p));return p


def load_pinned(path,expected,name):
    # Execute exactly the checked buffer; never an ambient/cached .pyc.
    path=Path(path);body=path.read_bytes();import hashlib
    need(hashlib.sha256(body).hexdigest()==expected,'module bytes differ: '+str(path))
    module=types.ModuleType(name);module.__file__=str(path.resolve());sys.modules[name]=module
    try:exec(compile(body,str(path),'exec'),module.__dict__)
    except BaseException:sys.modules.pop(name,None);raise
    return module


def verify_qualified_sources(root=None):
    root=Path(root) if root is not None else HERE/'qualified007'
    found={}
    for name,expected in QUALIFIED_PINS.items():
        p=root/name;need(p.is_file() and not p.is_symlink(),'qualified source missing or symlink: '+name)
        need(sha(p)==expected,'qualified source differs: '+name);found[name]=expected
    return found


def validate_intent(intent):
    need(type(intent)is dict and intent.get('schema')=='z1_screen_cpu32_intent_v1','explicit pilot CPU intent required')
    need(intent.get('status')=='ADMITTED_ONCE' and intent.get('scope')=='screen_science32','root admitted pilot-science intent required')
    need(type(intent.get('attempts'))is int and intent['attempts']==32,'exact32 scientific starts required')
    need(intent.get('cases')==['pilot-bank32'] and intent.get('device')=='cpu','one pilot bank case on qualified CPU only')
    need(intent.get('retry_allowed')is False and intent.get('resume_allowed')is False,'retry/resume forbidden')
    validate_record(intent.get('policy'))
    for key in ('reset_bank','bank_adapter'):
        v=intent.get(key);need(type(v)is dict and set(v)=={'path','sha256'},'bound '+key+' required')
        need(Path(v['path']).is_absolute() and type(v['sha256'])is str and len(v['sha256'])==64,'absolute SHA reference required')
    need(type(intent.get('evaluation_seed'))is int and 0<intent['evaluation_seed']<2**31,'explicit fresh evaluation seed required')
    need(type(intent.get('max_output_bytes'))is int and 0<intent['max_output_bytes']<=134217728,'bounded128MiB output cap required')
    need(type(intent.get('deadline_epoch'))in(int,float) and intent['deadline_epoch']>0,'finite absolute deadline required')
    canonical(intent)
    return intent


def install_verified_observer(adapter,obs,bank_evidence):
    """Qualified after_reset observer glue, replacing only scalar-route guard.

    The balanced bank's full qpos/qvel/route/reference verification has already
    succeeded. Observer class, installation, canonical mapping and output shape
    are byte-identical qualified007 components; no measurement math is copied.
    """
    from safety_observer import SafetyAccumulator
    durable(adapter.output/'reset-observed.json',dict(case_id='pilot-bank32',worlds=list(range(adapter.env.num_envs)),geometry={'reset_bank':adapter.intent['reset_bank'],'actual_reset_evidence':'pilot-reset-bank-evidence.json'},observation_sha256=adapter.tensor_hash(dict(obs)),actor_sha256=adapter.actor_before,normalizer_sha256=adapter.normalizer_before))
    adapter.observer=SafetyAccumulator(adapter.env.num_envs,adapter.env.device,adapter.env.physics_dt,native_window_s=.2)
    adapter.stack.enter_context(adapter.observer.install(adapter.env))
    path=adapter.output/'observer-mapping.json';durable(path,adapter.observer.mapping_record)
    adapter.mapping_binding={'path':str(path.resolve()),'sha256':sha(path)}


def validate_final_contract(adapter):
    from src.tasks.hammer.rl.impact_speed_contract import build_impact_speed_metadata,validate_impact_speed_metadata,validate_impact_speed_models
    live=build_impact_speed_metadata(adapter.env,adapter.runner.env.clip_actions)
    expected=adapter.pilot_record['impact_speed_contract']
    validate_impact_speed_metadata(expected,live)
    validate_impact_speed_models(adapter.runner.alg.actor,adapter.runner.alg.critic,live)


def make_adapter_class(qualified_base):
    class PilotHorizonAdapter(qualified_base):
        def __init__(self,intent,output,bank_module):
            # Native measurement expects only native_window_s; retain the entire
            # actual final policy record, adding this MEASUREMENT envelope field.
            measurement=dict(intent,policy=dict(intent['policy'],native_window_s=.2))
            super().__init__(measurement,('pilot-bank32',1,0.),output)
            self.pilot_record=intent['policy'];self.bank_module=bank_module
            self.pilot_balanced_bank=True;self.expected_route_signs=(-1,0,1)

        def bind(self,env,runner,saved,contract):
            validate_record(self.pilot_record)
            need(canonical(saved)==canonical(self.pilot_record['impact_speed_contract']),'saved contract differs from admitted full screen contract')
            need(canonical(saved)==canonical(contract),'live/saved metadata differ including full symmetric plant')
            from src.tasks.hammer.rl.impact_speed_contract import validate_impact_speed_models
            validate_impact_speed_models(runner.alg.actor,runner.alg.critic,contract)
            need(env.num_envs==32 and str(env.device)=='cpu','actual environment must be CPU32')
            need(type(env.common_step_counter)is int and env.common_step_counter==2400,'actual loaded common-step counter differs')
            super().bind(env,runner,saved,contract)

        def __enter__(self):
            # New pilot branch only: bank owns balanced per-world route delivery.
            # The qualified base's forced scalar+1 route is intentionally not used.
            cfg=self.env.event_manager.get_term_cfg('sample_strike_route_signs')
            self.route_cfg=cfg;self.original_route=cfg.func
            try:
                ref=self.intent['reset_bank']
                self.bank_context=self.stack.enter_context(self.bank_module.create_context(self.env,ref['path'],ref['sha256']))
                bank_sampler=cfg.func
                def observe_route(source,env_ids,**params):
                    result=bank_sampler(source,env_ids,**params)
                    worlds=env_ids.detach().cpu().tolist()
                    self.pending_reset_worlds.update(worlds)
                    durable(self.output/f'reset-event-{self.reset_event_count:05d}.json',
                        {'worlds':worlds,'phase':'native route reset event; full reset completion not yet established'})
                    self.reset_event_count+=1
                    return result
                self.route_wrapper=observe_route;cfg.func=observe_route
                # Restore the bank-owned wrapper first, then bank context restores
                # both original native events. Base __exit__ sees original_route.
                self.stack.callback(setattr,cfg,'func',bank_sampler)
                return self
            except BaseException:
                self.stack.close();raise

        def after_reset(self,obs):
            ref=self.intent['reset_bank']
            evidence=self.bank_module.verify_reset(self.env,obs,ref['path'],ref['sha256'])
            need(type(evidence)is dict and bool(evidence),'verified full realized bank evidence required before observation')
            durable(self.output/'pilot-reset-bank-evidence.json',evidence)
            install_verified_observer(self,obs,evidence)

        def __exit__(self,*exc):
            self.final_contract_restored=False
            result=super().__exit__(*exc)
            if not exc or exc[0] is None:
                validate_final_contract(self)
                self.final_contract_restored=True
            return result
    return PilotHorizonAdapter


def registered_screen_env_equal(registered,expected):
    """Compare all config fields, normalizing only the proven terrain default.

    EntityCfg's default factory creates a fresh no-closure spec lambda for each
    config. Registry deepcopy preserves that function identity. Neither lambda
    is invoked here; only the fresh comparison config is temporarily changed.
    """
    try:
        a=registered.scene.terrain.spec_fn;b=expected.scene.terrain.spec_fn
    except AttributeError:return False
    if not (type(a)is types.FunctionType and type(b)is types.FunctionType
            and a.__code__ is b.__code__ and a.__globals__ is b.__globals__
            and a.__module__==b.__module__=='mjlab.entity.entity'
            and a.__qualname__==b.__qualname__=='EntityCfg.<lambda>.<locals>.<lambda>'
            and all(getattr(f,key)is None for f in (a,b)for key in ('__defaults__','__kwdefaults__','__closure__'))):return False
    filename=a.__globals__.get('__file__')
    if type(filename)is not str or Path(a.__code__.co_filename).resolve()!=Path(filename).resolve():return False
    try:
        expected.scene.terrain.spec_fn=a
        return registered==expected
    finally:expected.scene.terrain.spec_fn=b


def register_screen_tasks():
    """Validate native import-time registration; never register a second time.

    Frozen src.tasks recursively imports config.z1, whose module-level call
    owns all eight registrations. Complete source/origin checks and the ambient
    module refusal precede this seam. Compare every registered configuration
    and runner against those same frozen factories, without an environment.
    """
    from mjlab.tasks.registry import list_tasks,load_env_cfg,load_rl_cfg,load_runner_cls
    from src.tasks.hammer.config.z1.spline_credit_critic_cfg import TASKS,credit_critic_env_cfg,credit_critic_rl_cfg
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    from pilot_contract import BASE
    expected={(deadline,mode):BASE+f'-Credit{deadline}-Critic{mode.title()}'
              for deadline in (50,100,150,200)for mode in ('current','event')}
    need(TASKS==expected,'frozen screen task mapping differs')
    registered=list_tasks();need(len(registered)==len(set(registered)),'duplicate registered task identities')
    actual={task for task in registered if task.startswith(BASE+'-Credit')}
    need(actual==set(expected.values()),'complete exact eight native screen registrations required')
    for (deadline,mode),task in expected.items():
        for play in (False,True):
            need(registered_screen_env_equal(load_env_cfg(task,play=play),credit_critic_env_cfg(play=play,credit_deadline_ms=deadline,critic_mode=mode)),'registered screen environment differs: '+task)
        need(load_rl_cfg(task)==credit_critic_rl_cfg(critic_mode=mode),'registered screen RL factory differs: '+task)
        need(load_runner_cls(task)is HammerOnPolicyRunner,'registered screen runner differs: '+task)



def adapt_pilot_condition(evaluator,source_path):
    """One counted opt-in post-rollout stratum seam; qualified bytes stay intact.

    No environment, policy or measurement is called by this derivation. Legacy
    adapters retain their scalar case contract; only our balanced pilot marker
    selects its explicit three-route tuple.
    """
    need(not hasattr(evaluator,'pilot_condition_adaptation_counts'),'pilot condition already adapted')
    path=Path(source_path);body=path.read_bytes();import hashlib
    need(hashlib.sha256(body).hexdigest()==QUALIFIED_PINS['current_evaluator_candidate.py'],'qualified evaluator source drift')
    nodes=[n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef)and n.name=='_condition']
    need(len(nodes)==1,'one qualified condition function required');node=nodes[0]
    old=ast.parse('(adapter.case[1],) if adapter is not None else (-1,0,1) if fixed_sign is None else (fixed_sign,)',mode='eval').body
    count=0
    for assignment in ast.walk(node):
        if (isinstance(assignment,ast.Assign)and len(assignment.targets)==1 and isinstance(assignment.targets[0],ast.Name)
            and assignment.targets[0].id=='expected_signs'and ast.dump(assignment.value,include_attributes=False)==ast.dump(old,include_attributes=False)):
            replacement=ast.parse('(adapter.expected_route_signs if getattr(adapter,"pilot_balanced_bank",False) is True else (adapter.case[1],)) if adapter is not None else (-1,0,1) if fixed_sign is None else (fixed_sign,)',mode='eval').body
            assignment.value=ast.copy_location(replacement,assignment.value);count+=1
    need(count==1,'qualified route expectation seam changed')
    tree=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(tree,str(path)+':pilot_balanced_route_expectation','exec'),evaluator.__dict__)
    evaluator.pilot_condition_source=ast.unparse(tree)
    evaluator.pilot_condition_adaptation_counts={'balanced_route_expected_signs':count}
    return evaluator


def prepare_pilot_case(evaluator,qualified_base,intent,root,bank):
    """Validate real CLI while the case path is fresh, then exclusively create."""
    output=Path(root)/'pilot-bank32'
    args=evaluator.parse_args(evaluator_arguments(intent['policy'],output=str(output),seed=intent['evaluation_seed']))
    output.mkdir(exist_ok=False)
    adapter=make_adapter_class(qualified_base)(intent,output,bank);args.horizon_adapter=adapter
    return args,adapter,output

def execute_pilot_intent(intent):
    """Actual finite entry; caller must supply its live admitted supervisor.

    Never call this for a pure test or zero-step probe. Root owns the worker,
    physical claim, source/runtime origin audit and process cleanup contract.
    """
    validate_intent(intent)
    need(time.time()<intent['deadline_epoch'],'admitted deadline already expired')
    q=HERE/'qualified007';verify_qualified_sources(q)
    need(not any(n=='src' or n.startswith('src.') or n=='scripts' or n.startswith('scripts.') for n in sys.modules),'ambient training/evaluator modules loaded before frozen imports')
    source_path=Path(intent['source_root']);need(not source_path.is_symlink(),'source root symlink forbidden');source=source_path.resolve(strict=True)
    expected={reference(r).resolve():r['sha256'] for r in intent['source_files']}
    actual={p.resolve() for folder in ('src','scripts') for p in (source/folder).rglob('*.py')}
    need(bool(actual) and actual.issubset(expected),'complete sealed source Python inventory required')
    packages={reference(r).resolve():r['sha256'] for r in intent['package_files']}
    for name,pin in QUALIFIED_PINS.items():need(packages.get((q/name).resolve())==pin,'qualified package inventory missing: '+name)
    origin=json.loads((q/'current-evaluator-origin.json').read_text())
    need(sha(source/'scripts/evaluate_impact_speed.py')==origin['sha256'],'qualified evaluator origin differs')
    bank_path=reference(intent['bank_adapter']);reference(intent['reset_bank'])
    need(packages.get(bank_path.resolve())==intent['bank_adapter']['sha256'],'bank adapter absent from sealed package closure')
    root=Path(intent['output_root']).resolve(strict=True)
    need(root.is_dir() and not root.is_symlink(),'fresh bound output directory required')
    sys.path[:0]=[str(q),str(source),str(bank_path.parent)]
    import torch
    runtime=dict(python=sys.version,executable=sys.executable,platform=sys.platform,packages={n:importlib.metadata.version(n) for n in ('torch','mujoco','mujoco-warp','mjlab')},torch_cuda=torch.version.cuda,cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    need(runtime==intent['runtime_identity'],'actual pinned CPU software runtime differs')
    durable(root/'actual-runtime.json',runtime)
    import mjlab.tasks
    import src.tasks
    need(Path(sys.modules['src.tasks'].__file__).resolve().is_relative_to(source),'task import origin differs')
    register_screen_tasks()
    checkpoint=Path(intent['policy']['checkpoint_path'])
    need(sha(checkpoint)==intent['policy']['checkpoint_sha256'],'checkpoint bytes changed before load')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    durable(root/'pilot-checkpoint-audit.json',validate_payload(payload,intent['policy']))
    del payload
    qualified=load_pinned(q/'horizon_integration.py',QUALIFIED_PINS['horizon_integration.py'],'pilot_qualified_integration007')
    evaluator=load_pinned(q/'current_evaluator_candidate.py',QUALIFIED_PINS['current_evaluator_candidate.py'],'pilot_qualified_evaluator007')
    bank=load_pinned(bank_path,intent['bank_adapter']['sha256'],'pilot_bound_bank_adapter')
    need(callable(getattr(bank,'create_context',None)) and callable(getattr(bank,'verify_reset',None)),'bank event/verification interface required')
    adapt_pilot_condition(evaluator,q/'current_evaluator_candidate.py')
    args,adapter,output=prepare_pilot_case(evaluator,qualified.HorizonAdapter,intent,root,bank)
    rows,report=evaluator._condition(args,None,0)
    need(adapter.started_counts=={w:1 for w in range(32)},'actual physical first-start population differs')
    need(adapter.final_contract_restored is True,'final full screen contract restoration not verified')
    for item in [*rows,report]:
        item.update(observer_mapping=adapter.mapping_binding,native_onset_trace=adapter.trace_binding,safety_endpoint_summary=adapter.endpoint_binding,reset_bank=intent['reset_bank'])
    qualified.durable(output/'episodes.json',rows);qualified.durable(output/'report.json',report)
    need(sha(checkpoint)==intent['policy']['checkpoint_sha256'],'checkpoint changed during evaluation')
    reference(intent['reset_bank']);reference(intent['bank_adapter']);verify_qualified_sources(q)
    return dict(status='RAW_REVIEW_REQUIRED',attempts=32,reports=[dict(case_id='pilot-bank32',report_sha256=sha(output/'report.json'),observer_mapping=adapter.mapping_binding,native_onset_trace=adapter.trace_binding,safety_endpoint_summary=adapter.endpoint_binding)],observer_mappings=[adapter.mapping_binding],native_onset_traces=[adapter.trace_binding],safety_endpoint_summaries=[adapter.endpoint_binding],checkpoint_contract_validated_before_overlay=True,final_contract_restored=True,reset_frames_sampled=0,scientific_qualification=False,reset_bank=intent['reset_bank'],screen_policy_id=intent['policy']['policy_id'])
