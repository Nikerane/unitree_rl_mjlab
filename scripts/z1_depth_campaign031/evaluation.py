"""031 checkpoint bindings around unchanged native32/contact801 arithmetic."""
import ast
import copy
import hashlib
import importlib
import itertools
import json
import math
from collections.abc import Mapping
from pathlib import Path
import sys
import time
import types
from .files import canonical, digest, durable_reference, inventory, need, read_bytes, read_json, reference, sha_file, verify_ref, write_once
from .manifest import LIMITS, training_cell

HERE=Path(__file__).resolve().parent
VENDOR=HERE/'vendor'
BANK_SHA='b0baebc9dee774c89091a16ed7ba6ce5bfb3df4e4a639b6faabf56ec9da43f22'
BANK_ADAPTER_SHA='63cb613f85f7e3121690fd15b56eb3c3816ad3ac03c154e0385cb9010ab44379'
EVALUATION_SEED=2026103001
_READER_IDS=itertools.count()


def verify_copies():
    pins=read_json(HERE/'COPIED-SOURCES031.json')
    need(len(pins)==25 and sum(r['size'] for r in pins.values())==413171,'closed vendor size/count differs')
    need(inventory(VENDOR,413171)==pins,'vendor byte closure differs')
    return {p:r['sha256'] for p,r in pins.items()}


def load_pinned(path, expected, name):
    body=read_bytes(path);need(hashlib.sha256(body).hexdigest()==expected,'vendor SHA differs')
    need(name not in sys.modules,'ambient vendor module refused')
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    try:exec(compile(body,str(path),'exec'),module.__dict__)
    except BaseException:
        sys.modules.pop(name,None);raise
    return module


def pilot_utilities():
    pins=verify_copies()
    for name in ('pilot_contract','pilot_integration'):
        rel='pilot-compat001/'+name+'.py';module=load_pinned(VENDOR/rel,pins[rel],name)
    return module


def validate_contract031(contract, index):
    marker=read_json(HERE/'identities031.json')[training_cell(index)['arm']]
    need(canonical(contract.get('spline_depth_reward031'))==canonical(marker),'selected031 marker differs')
    need(canonical(contract.get('spline_split_credit'))==canonical(marker['baseline_split']),'baseline split differs')
    expected=dict(schema_version='impact_speed_v1',action_type='joint_position_variable_impedance_sampled_pd',
        controller_identity='sampled_vic500_v1',action_dim=13,physics_dt_s=.00025,control_decimation=80,
        speed_bounds_mps=[1.,1.5],speed_mode='policy',observation_widths={'actor':43,'critic':43})
    need(all(canonical(contract.get(k))==canonical(v) for k,v in expected.items()),'physical031 contract differs')
    safety=dict(use_vel=True,use_impulse=True,imp_limit=[.369,.246,.738,.369,.246,.0164],imp_max_p=.2,imp_seed=.001,
        limit=3.1415,max_p=.5,min_p=0.,tau=.95,vel_detection='substep')
    need(canonical(contract.get('safety',{}).get('cat_soft'))==canonical(safety),'unchanged safety differs')
    return contract


def validate_initial(initial,index):
    cell=training_cell(index)
    need(initial.get('cell')==cell and type(initial.get('actual_counter')) is int and initial['actual_counter']==0
         and initial.get('phase')=='BEFORE_UPDATE_0_AND_FIRST_ROLLOUT'
         and initial.get('model_initialization_seed')==cell['seed'],'fresh initialized state identity differs')
    validate_contract031(initial['contract'],index)
    std=initial.get('initial_std')
    need(type(std) is list and len(std)==13 and all(type(v) in (int,float) and math.isclose(v,t,rel_tol=1e-6,abs_tol=1e-8)
         for v,t in zip(std,[.3]*12+[1.])),'initial031 exploration std differs')
    hashes=initial.get('hashes',{})
    fields=('actor','critic','actor_shared_non_std','shared_critic43','actor_normalizer','critic_normalizer',
        'collection_torch_rng','collection_python_rng','collection_numpy_rng','collection_cuda_rng')
    import re
    need(all(type(hashes.get(k)) is str and re.fullmatch('[0-9a-f]{64}',hashes[k]) for k in fields),'complete model/normalizer/RNG fingerprints required')
    observed=initial.get('initial_observations',{})
    need(observed.get('actor_shape')==[1024,43] and observed.get('critic_shape')==[1024,43]
         and all(type(observed.get(k)) is str and re.fullmatch('[0-9a-f]{64}',observed[k]) for k in ('actor_sha256','critic_shared43_sha256')),
         'saved common initial observations differ')
    return initial


def checkpoint_seal(manifest, index, label, output=None):
    need(type(label) is int and label in (0,24,49,99,249,499),'selected saved label required')
    training_cell(index)
    root=Path(manifest['cells'][index]['output'] if output is None else output)
    initial=read_json(root/'initial-state.json')
    need(initial.get('cell')==training_cell(index) and initial.get('actual_counter')==0
         and initial.get('phase')=='BEFORE_UPDATE_0_AND_FIRST_ROLLOUT','fresh initial031 state required')
    validate_initial(initial,index)
    checkpoint=root/'train'/f'model_{label}.pt'
    need(checkpoint.is_file(),'selected checkpoint not yet sealed')
    live=read_json(root/f'checkpoint-{label}-live.json')
    need(type(live.get('iteration')) is int and live['iteration']==label and type(live.get('common_step_counter')) is int
         and live['common_step_counter']==(label+1)*24 and live.get('checkpoint_sha256')==sha_file(checkpoint),'checkpoint/live clock/hash differs')
    return dict(schema='z1_depth_checkpoint_seal031',manifest_sha256=digest(manifest),cell=index,
        checkpoint_label=label,counter=(label+1)*24,checkpoint=durable_reference(checkpoint),
        initial=durable_reference(root/'initial-state.json'),live=durable_reference(root/f'checkpoint-{label}-live.json'),
        impact_speed_contract=initial['contract'])


def _publish_seal(path,expected):
    """Only seals are idempotent: authenticate the exact immutable race winner.

    Claims/intents/submissions continue to use strictly exclusive write_once.
    """
    if not path.exists():
        try:write_once(path,expected)
        except FileExistsError:pass  # A competing sealed publication must match below.
    before=reference(path)
    need(canonical(read_json(path))==canonical(expected),'conflicting immutable seal: '+str(path))
    for key in ('initial','claim','checkpoint','live'):
        if key in expected:verify_ref(expected[key])
    after=durable_reference(path)
    need(after==before,'seal changed during winner authentication')
    return after


def publish_seals(manifest, index):
    """Publish complete live receipts; concurrent identical publishers agree."""
    root=Path(manifest['cells'][index]['output']);results=[]
    if not (root/'initial-state.json').is_file():return results
    try:initial=read_json(root/'initial-state.json')
    except (json.JSONDecodeError,UnicodeDecodeError):return results
    need(initial['cell']==training_cell(index) and initial['actual_counter']==0,'initial identity differs')
    validate_initial(initial,index)
    claim=reference(Path(manifest['root'])/'claims'/f'cell-{index:02d}.json')
    _publish_seal(root/'initial-seal.json',dict(schema='z1_depth_initial_seal031',manifest_sha256=digest(manifest),
        cell=index,initial=durable_reference(root/'initial-state.json'),claim=claim))
    for label in (99,499):
        if not (root/f'checkpoint-{label}-live.json').is_file():continue
        try:seal=checkpoint_seal(manifest,index,label)
        except (json.JSONDecodeError,UnicodeDecodeError):continue
        seal['claim']=claim
        results.append(_publish_seal(root/f'model-{label}-seal.json',seal))
    return results


def initial_anchor(manifest,index):
    need(index in (0,4,8),'A0 anchor required')
    root=Path(manifest['cells'][index]['output']);seal=read_json(root/'initial-seal.json')
    need(seal.get('schema')=='z1_depth_initial_seal031' and seal.get('manifest_sha256')==digest(manifest)
         and seal.get('cell')==index,'anchor lineage differs')
    verify_ref(seal['initial'],root/'initial-state.json');verify_ref(seal['claim'],Path(manifest['root'])/'claims'/f'cell-{index:02d}.json')
    claim=read_json(seal['claim']['path']);need(claim['manifest_sha256']==digest(manifest) and claim['kind']=='training' and claim['index']==index,'anchor consumed claim differs')
    initial=read_json(seal['initial']['path'])
    need(initial['cell']==training_cell(index) and initial['actual_counter']==0,'anchor initial identity differs')
    validate_initial(initial,index)
    return initial


def record_for_binding(manifest, case_index, seal):
    need(type(case_index) is int and 0<=case_index<24,'case0..23 required')
    d=manifest['cases'][case_index];index=d['cell']
    need(seal.get('schema')=='z1_depth_checkpoint_seal031' and seal.get('manifest_sha256')==digest(manifest)
         and seal['cell']==index and seal['checkpoint_label']==d['label'] and seal['counter']==d['counter'],'selected checkpoint seal differs')
    for key in ('checkpoint','initial','live','claim'):verify_ref(seal[key])
    need(seal['checkpoint']['path']==manifest['cells'][index]['output']+f"/train/model_{d['label']}.pt",'selected checkpoint path differs')
    validate_contract031(seal['impact_speed_contract'],index)
    return dict(policy_id=f"cell-{index:02d}-model-{d['label']}",case_index=case_index,
        training_cell=training_cell(index),checkpoint_label=d['label'],completed_updates=d['label']+1,
        common_step_counter=d['counter'],task='Unitree-Z1-Hammer-Spline-DepthReward031-'+training_cell(index)['arm'],
        checkpoint_path=seal['checkpoint']['path'],checkpoint_sha256=seal['checkpoint']['sha256'],
        impact_speed_contract=seal['impact_speed_contract'])


def validate_record(record):
    index=record['training_cell']['index'];need(record['training_cell']==training_cell(index),'record cell differs')
    need(record['checkpoint_label'] in (99,499) and record['common_step_counter']==(record['checkpoint_label']+1)*24,'record checkpoint clock differs')
    need(record['task']=='Unitree-Z1-Hammer-Spline-DepthReward031-'+training_cell(index)['arm'],'old/wrong task refused')
    validate_contract031(record['impact_speed_contract'],index)
    return record


# Narrow unchanged tensor/payload checks from audited029 records (no authority).
def finite_tree(value):
    if hasattr(value,'detach'):
        need(bool(value.detach().isfinite().all().item()), 'nonfinite checkpoint tensor')
    elif isinstance(value,float):
        need(math.isfinite(value), 'nonfinite checkpoint scalar')
    elif isinstance(value,Mapping):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value,(tuple,list)):
        for item in value:
            finite_tree(item)


def expected_state_shapes(width, actor=False):
    need(type(width) is int and width in (43,47) and (not actor or width == 43), 'exact actor43/current43/event47 width required')
    shapes = {f'obs_normalizer.physical.{key}':(1,40) for key in ('_mean','_var','_std')}
    shapes['obs_normalizer.physical.count'] = ()
    for layer,left,right in ((0,width,256),(2,256,128),(4,128,64),(6,64,13 if actor else 1)):
        shapes[f'mlp.{layer}.weight'] = (right,left); shapes[f'mlp.{layer}.bias'] = (right,)
    if actor:
        shapes['distribution.std_param'] = (13,)
    return shapes


def validate_state_shapes(state, width, actor=False):
    shapes = expected_state_shapes(width,actor)
    need(isinstance(state,Mapping) and set(state) == set(shapes), 'exact actor/critic tensor keys required')
    for key,shape in shapes.items():
        tensor = state[key]
        need(hasattr(tensor,'detach') and tuple(getattr(tensor,'shape',())) == shape, 'saved tensor shape differs: '+key)
    finite_tree(state)


def validate_payload(payload, record):
    validate_record(record)
    need(isinstance(payload,Mapping), 'checkpoint Mapping required')
    need(type(payload.get('iter')) is int and payload['iter'] == record['checkpoint_label'], 'actual checkpoint iteration differs')
    infos = payload.get('infos'); need(isinstance(infos,Mapping), 'checkpoint infos Mapping required')
    state = infos.get('env_state'); need(isinstance(state,Mapping), 'checkpoint env state required')
    need(type(state.get('common_step_counter')) is int and state['common_step_counter'] == record['common_step_counter'], 'loaded checkpoint clock differs')
    need(infos.get('cat_pressure_treatment') is None, 'pressure scheduler forbidden')
    need(canonical(infos.get('impact_speed_contract')) == canonical(record['impact_speed_contract']), 'full saved contract differs from producer')
    validate_state_shapes(payload.get('actor_state_dict'),43,True)
    validate_state_shapes(payload.get('critic_state_dict'),record['impact_speed_contract']['observation_widths']['critic'])
    finite_tree(payload)
    return dict(status='PASS_PAYLOAD_ONLY_NOT_AUTHORITY',iteration=record['checkpoint_label'],
                common_step_counter=record['common_step_counter'],checkpoint_sha256=record['checkpoint_sha256'])




# Audited configuration-only AST seam and inherited native adapter bind.
def make_adapter_class(qualified_base, pilot_utils):
    pilot_base = pilot_utils.make_adapter_class(qualified_base)
    class FamilyHorizonAdapter(pilot_base):
        def bind(self, env, runner, saved, contract):
            validate_record(self.pilot_record)
            need(canonical(saved) == canonical(self.pilot_record['impact_speed_contract']) == canonical(contract),
                 'full producer/saved/live contract differs')
            from src.tasks.hammer.rl.impact_speed_contract import validate_impact_speed_models
            validate_impact_speed_models(runner.alg.actor,runner.alg.critic,contract)
            need(type(env.num_envs) is int and env.num_envs == 32 and str(env.device) == 'cpu', 'CPU32 environment required')
            need(type(env.common_step_counter) is int and env.common_step_counter == self.pilot_record['common_step_counter'],
                 'actual selected checkpoint counter differs')
            # Intentionally bypass legacy PilotHorizonAdapter.bind's fixed2400 rule.
            qualified_base.bind(self,env,runner,saved,contract)
    return FamilyHorizonAdapter


def adapt_family_configuration(evaluator, source_path, config_factory):
    """Exactly one config assignment seam; no physics/control/math replacement."""
    need(not hasattr(evaluator,'thirty_configuration_adaptation_counts'), 'family configuration already adapted')
    pins = verify_copies(); path = Path(source_path)
    need(sha_file(path) == pins['pilot-compat001/qualified007/current_evaluator_candidate.py'], 'qualified evaluator source drift')
    if hasattr(evaluator,'pilot_condition_source'):
        need(evaluator.pilot_condition_adaptation_counts == {'balanced_route_expected_signs':1}, 'balanced route seam differs')
        tree = ast.parse(evaluator.pilot_condition_source)
    else:
        tree = ast.parse(path.read_bytes())
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == '_condition')
    old = ast.parse('cfg, agent_cfg = load_env_cfg(args.task, play=False), load_rl_cfg(args.task)').body[0]
    count = 0
    for n in ast.walk(node):
        if isinstance(n,ast.Assign) and ast.dump(n,include_attributes=False) == ast.dump(old,include_attributes=False):
            n.value = ast.copy_location(ast.parse('_thirty_config(args.task)',mode='eval').body,n.value); count += 1
    need(count == 1, 'training config assignment seam differs')
    tree = ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    evaluator._thirty_config = config_factory
    exec(compile(tree,str(path)+':thirty_config_only','exec'),evaluator.__dict__)
    evaluator.thirty_condition_source = ast.unparse(tree)
    evaluator.thirty_configuration_adaptation_counts = {'training_factory_config':count}



def selected_configuration(record,pilot_utils):
    validate_record(record)
    from scripts.z1_depth_reward_training031 import build_train_config
    from src.tasks.hammer.config.z1.spline_depth_reward031_cfg import depth_env_cfg
    from src.tasks.hammer.config.z1.spline_split_credit_cfg import split_rl_cfg
    from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    cell=record['training_cell'];task,cfg=build_train_config(cell['index'])
    need(task==record['task'] and pilot_utils.registered_screen_env_equal(load_env_cfg(task,play=False),depth_env_cfg(play=False,arm=cell['arm']))
         and load_rl_cfg(task)==split_rl_cfg(setting='vic_baseline') and load_runner_cls(task) is HammerOnPolicyRunner,'registry/factory differs')
    need(cfg.env.scene.num_envs==1024 and cfg.env.seed==cfg.agent.seed==cell['seed'] and cfg.agent.max_iterations==500
         and cfg.agent.resume is False and cfg.sampled_pd500 is True and cfg.sampled_pd500_gain_mode=='variable','original031 factory differs')
    called=[]
    def selected(task_name):
        need(task_name==task and not called,'one config seam required');called.append(True)
        return copy.deepcopy(cfg.env),copy.deepcopy(cfg.agent)
    return selected


def measurement_context(intent):
    manifest=intent['manifest']
    return dict(policy=intent['record'],output_root=intent['output'],source_root=manifest['source_root'],
        source_files=[dict(path=str(Path(manifest['source_root'])/p),sha256=sha) for p,sha in manifest['source_files'].items()],
        reset_bank=dict(path=str(VENDOR/'reset-bank001/BANK32-BALANCED-PROSPECTIVE002.json'),sha256=BANK_SHA),
        bank_adapter=dict(path=str(VENDOR/'reset-bank001/reset_bank.py'),sha256=BANK_ADAPTER_SHA),
        evaluation_seed=EVALUATION_SEED,max_output_bytes=LIMITS['eval_output'],deadline_epoch=intent['deadline_epoch'])


def execute(intent):
    """Called only by031 owned child after source/runtime/compiler admission."""
    from .worker import verify_import_origins
    output=Path(intent['output']);manifest=intent['manifest'];record=intent['record']
    pins=verify_copies();pilot=pilot_utilities();q=VENDOR/'pilot-compat001/qualified007';context=measurement_context(intent)
    sys.path[:0]=[str(q),str(VENDOR/'reset-bank001')]
    import torch
    need(not torch.cuda.is_available(),'CPU evaluation must hide CUDA')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    verify_import_origins(manifest)
    payload=torch.load(record['checkpoint_path'],map_location='cpu',weights_only=False)
    write_once(output/'checkpoint-audit.json',validate_payload(payload,record));del payload
    factory=selected_configuration(record,pilot)
    qualified=load_pinned(q/'horizon_integration.py',pins['pilot-compat001/qualified007/horizon_integration.py'],'depth031_horizon')
    evaluator=load_pinned(q/'current_evaluator_candidate.py',pins['pilot-compat001/qualified007/current_evaluator_candidate.py'],'depth031_evaluator')
    bank=load_pinned(VENDOR/'reset-bank001/reset_bank.py',BANK_ADAPTER_SHA,'depth031_bank')
    origin=read_json(q/'current-evaluator-origin.json')
    need(sha_file(Path(manifest['source_root'])/'scripts/evaluate_impact_speed.py')==origin['sha256'],'measured scientific origin differs')
    pilot.adapt_pilot_condition(evaluator,q/'current_evaluator_candidate.py')
    adapt_family_configuration(evaluator,q/'current_evaluator_candidate.py',factory)
    case=output/'pilot-bank32';case.mkdir(exist_ok=False)
    args=evaluator.parse_args(['--task',record['task'],'--checkpoint',record['checkpoint_path'],
        '--speed-bounds','1.0','1.5','--speed-cost','0.8493676','--seed',str(EVALUATION_SEED),
        '--num-envs','32','--mode','policy','--output',str(case),'--device','cpu','--sampled-pd500',
        '--sampled-pd500-gain-mode','variable','--measurement-protocol','native'])
    adapter=make_adapter_class(qualified.HorizonAdapter,pilot)(context,case,bank);args.horizon_adapter=adapter
    need(time.time()<intent['deadline_epoch'],'expired before evaluation construction')
    start=time.time();rows,report=evaluator._condition(args,None,0);finish=time.time()
    need(finish<intent['deadline_epoch'] and adapter.started_counts=={w:1 for w in range(32)} and adapter.final_contract_restored is True,'native32 completion differs')
    for item in [*rows,report]:
        item.update(observer_mapping=adapter.mapping_binding,native_onset_trace=adapter.trace_binding,
                    safety_endpoint_summary=adapter.endpoint_binding,reset_bank=context['reset_bank'])
    qualified.durable(case/'episodes.json',rows);qualified.durable(case/'report.json',report)
    verify_import_origins(manifest);verify_copies()
    result=dict(status='RAW_REVIEW_REQUIRED',attempts=32,reports=[dict(case_id='pilot-bank32',report_sha256=sha_file(case/'report.json'),
        observer_mapping=adapter.mapping_binding,native_onset_trace=adapter.trace_binding,safety_endpoint_summary=adapter.endpoint_binding)],
        observer_mappings=[adapter.mapping_binding],native_onset_traces=[adapter.trace_binding],safety_endpoint_summaries=[adapter.endpoint_binding],
        checkpoint_contract_validated_before_overlay=True,final_contract_restored=True,reset_frames_sampled=0,
        scientific_qualification=False,reset_bank=context['reset_bank'],policy_id=record['policy_id'],
        execution_timing=dict(started_epoch=start,finished_epoch=finish),configuration_adaptation_counts=evaluator.thirty_configuration_adaptation_counts,
        route_adaptation_counts=evaluator.pilot_condition_adaptation_counts)
    write_once(output/'worker-result.json',result)
    return result


def audit_case(intent):
    root=Path(intent['output']);raw=read_json(root/'worker-result.json');record=intent['record']
    need(raw['status']=='RAW_REVIEW_REQUIRED' and raw['attempts']==32 and raw['final_contract_restored'] is True
         and raw['reset_frames_sampled']==0 and raw['configuration_adaptation_counts']=={'training_factory_config':1}
         and raw['route_adaptation_counts']=={'balanced_route_expected_signs':1},'raw native32 result differs')
    need(intent['issued_epoch']<=raw['execution_timing']['started_epoch']<=raw['execution_timing']['finished_epoch']<intent['deadline_epoch'],'evaluation execution clock differs')
    files={name:ref for name,ref in inventory(root,LIMITS['eval_output']).items() if name not in ('stdout.log','stderr.log')};pins=verify_copies()
    reader=load_pinned(VENDOR/'sealer001/measurement_reader.py',pins['sealer001/measurement_reader.py'],'depth031_reader_'+str(next(_READER_IDS)))
    context=measurement_context(intent)
    measured=reader.audit_measurements(reader.dependencies(),context,root,read_json(context['reset_bank']['path']))
    need(measured.get('denominators',{}).get('all_first_episodes')==32,'exact32 denominator required')
    return dict(schema='z1_depth_measurement_audit031',status='PASS_FILE_AUDIT_ONLY',kind='evaluation',index=intent['index'],
        cell=record['training_cell'],record=record,binding=intent['binding'],reset_bank=context['reset_bank'],evaluation_seed=EVALUATION_SEED,
        manifest_sha256=digest(intent['manifest']),checkpoint_sha256=record['checkpoint_sha256'],
        claim_sha256=intent['claim']['sha256'],raw_files=files,measurement=measured)
