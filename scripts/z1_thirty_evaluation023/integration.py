"""Training-family preparation seams around unchanged qualified physical evaluator.

Only a fully checked, consumed T2 supervisor intent can reach actual execution.
This module is not a scheduler, process supervisor, authority writer or retry path.
"""
import ast
import copy
import importlib
import json
import math
from pathlib import Path
import re
import sys
import time
import types
from . import release as a
from .files import canonical, digest, need, read_json, sha_file, verify_ref, write_once
from .producer import validate_producer, verify_package, verify_frozen_files
from .qualified import VENDOR, load_pinned, pilot_utilities, verify_copies
from .records import record_for_binding, validate_record, validate_payload, evaluator_arguments

INTENT_FIELDS = {'schema','release','grant','resolved_ref','binding','record','index','output','private','claim_ref',
                 'parent','child_argv','slurm','nonce','issued_epoch','deadline_epoch','elapsed_limit_seconds'}


def validate_execution_intent(intent, *, now=None):
    """Closed metadata validation only; verified_inputs additionally reads all refs."""
    now = time.time() if now is None else now
    need(type(intent) is dict and set(intent) == INTENT_FIELDS, 'closed claimed evaluation intent required')
    need(intent['schema'] == 'z1_thirty_evaluation_intent023', 'new explicit evaluation intent schema required')
    a.validate_grant(intent['release'],intent['grant'],now=now)
    d = a.case_for_index(intent['index'])
    need(canonical(intent['binding'].get('descriptor')) == canonical(d), 'bound descriptor differs')
    validate_record(intent['record'])
    need(intent['record']['case_index'] == intent['index'] and intent['output'] == d['output']
         and intent['private'] == d['private'], 'case output/private/record differs')
    need(type(intent['nonce']) is str and re.fullmatch('[0-9a-f]{32}',intent['nonce']), 'unique case nonce required')
    need(all(type(intent[k]) in (int,float) and math.isfinite(intent[k]) for k in ('issued_epoch','deadline_epoch')),
         'finite case clocks required')
    need(intent['elapsed_limit_seconds'] == 7200 and type(intent['elapsed_limit_seconds']) is int
         and 0 <= intent['issued_epoch'] <= now < intent['deadline_epoch'] <= intent['grant']['expires_epoch']
         and intent['deadline_epoch']-intent['issued_epoch'] <= 7200, 'expired/overlong case clock')
    need(type(intent['parent']) is dict and type(intent['parent'].get('pid')) is int
         and intent['parent']['pid'] > 1 and type(intent['parent'].get('startticks')) is str
         and intent['parent']['startticks'].isdigit() and type(intent['parent'].get('argv')) is list
         and intent['parent']['argv'], 'parent PID/startticks/argv required')
    need(type(intent['child_argv']) is list and intent['child_argv'] and type(intent['slurm']) is dict
         and re.fullmatch('[1-9][0-9]*',intent['slurm'].get('job_id','')), 'exact child argv/Slurm required')
    return intent


def validate_historical_intent(intent):
    """Immutable provenance only: NEVER use this to admit or execute a case.

    File audits may occur after authority expiry. The original issue instant
    must have been valid; genuine consumed claim and execution times are joined
    separately by verified_inputs and the measurement wrapper.
    """
    need(type(intent) is dict and type(intent.get('issued_epoch')) in (int,float), 'saved issue clock required')
    return validate_execution_intent(intent,now=intent['issued_epoch'])


def verify_import_origins(producer):
    source = Path(producer['source_root'])
    for name,module in list(sys.modules.items()):
        if name == 'src' or name.startswith('src.') or name == 'scripts' or name.startswith('scripts.'):
            origin = getattr(module,'__file__',None)
            # scripts is a native namespace package in this repository.
            if origin is None:
                need(name == 'scripts' and set(map(str,getattr(module,'__path__',()))) == {str(source/'scripts')}, 'ambient namespace import refused')
                continue
            path = Path(origin)
            need(path.is_relative_to(source) and sha_file(path) == producer['source_files'].get(str(path.relative_to(source))),
                 'scientific import escaped frozen producer: '+name)


def validate_claim_record(intent, claim):
    required = dict(schema='z1_thirty_evaluation_claim023',status='CLAIMED_ONCE',release_sha256=digest(intent['release']),
                    grant_sha256=digest(intent['grant']),resolved_sha256=intent['resolved_ref']['sha256'],index=intent['index'],
                    nonce=intent['nonce'],parent=intent['parent'],slurm=intent['slurm'],issued_epoch=intent['issued_epoch'])
    need(all(canonical(claim.get(k)) == canonical(v) for k,v in required.items()), 'consumed claim lineage differs')


def verified_inputs(intent, *, historical=False):
    need(type(historical) is bool, 'explicit historical audit selector required')
    (validate_historical_intent if historical else validate_execution_intent)(intent)
    release = intent['release']; verify_package(release)
    path = Path(release['producer_manifest']['path'])
    need(sha_file(path) == a.PRODUCER_SHA, 'producer manifest bytes differ')
    producer = validate_producer(read_json(path)); verify_frozen_files(producer)
    verify_import_origins(producer)
    resolved_path = verify_ref(intent['resolved_ref'],expected_path=Path(a.ROOT)/'resolved.json',max_bytes=32*a.MIB)
    resolved = read_json(resolved_path,cap=32*a.MIB)
    need(resolved.get('schema') == 'z1_thirty_evaluation_resolved023'
         and resolved.get('release_sha256') == digest(release) and resolved.get('grant_sha256') == digest(intent['grant'])
         and resolved.get('producer_manifest_sha256') == a.PRODUCER_SHA
         and resolved.get('source_manifest_sha256') == digest(producer['source_files'])
         and type(resolved.get('cases')) is list and len(resolved['cases']) == 60, 'resolved lineage differs')
    binding = resolved['cases'][intent['index']]
    need(canonical(binding) == canonical(intent['binding']), 'selected resolved binding differs')
    need(canonical(record_for_binding(binding,resolved['source_manifest_sha256'])) == canonical(intent['record']), 'selected policy record differs')
    d = binding['descriptor']
    paths = {'checkpoint':d['checkpoint_path'],'live':d['checkpoint_live_receipt'],
             'initial':d['producer_case_output']+'/initial-state.json','completion':d['producer_cell_final'],'slot_final':d['producer_slot_final']}
    for key,target in paths.items():
        verify_ref(binding[key],expected_path=target)
    initial = read_json(binding['initial']['path'])
    need(initial.get('cell') == d['training_cell'] and type(initial.get('actual_counter')) is int and initial['actual_counter'] == 0
         and initial.get('phase') == 'BEFORE_UPDATE_0_AND_FIRST_ROLLOUT'
         and canonical(initial.get('contract')) == canonical(intent['record']['impact_speed_contract']), 'sealed initial contract/cell differs')
    live = read_json(binding['live']['path'])
    need(type(live.get('iteration')) is int and live['iteration'] == d['checkpoint_label']
         and type(live.get('common_step_counter')) is int and live['common_step_counter'] == d['expected_common_step_counter']
         and live.get('checkpoint_sha256') == binding['checkpoint']['sha256'], 'selected actual live checkpoint differs')
    complete = read_json(binding['completion']['path']); final = read_json(binding['slot_final']['path'])
    seals = resolved['cell_seals']; seal = seals.get(str(d['training_cell_index']),seals.get(d['training_cell_index']))
    need(canonical(complete) == canonical(dict(status='COMPLETE',index=d['training_cell_index'],manifest_sha256=a.PRODUCER_SHA,artifacts=seal)), 'producer completion lineage differs')
    slot = d['training_slot']
    need(final.get('status') == 'COMPLETE' and final.get('error') is None and final.get('slot') == slot
         and final.get('job_id') == a.PRODUCER_JOBS[slot]
         and canonical(final.get('artifacts')) == canonical(resolved['slot_seals'][slot]), 'producer slot final lineage differs')
    claim_path = verify_ref(intent['claim_ref'],expected_path=d['claim'],max_bytes=a.MIB)
    claim = read_json(claim_path,cap=a.MIB)
    validate_claim_record(intent,claim)
    verify_copies()
    return producer, resolved


def measurement_context(intent, producer):
    """New metadata wrapper; old schema/admission is deliberately never invoked."""
    root = Path(intent['release']['package_root'])/'vendor'
    return dict(policy=intent['record'],output_root=intent['output'],source_root=producer['source_root'],
        source_files=[{'path':str(Path(producer['source_root'])/p),'sha256':pin} for p,pin in producer['source_files'].items()],
        reset_bank={'path':str(root/'reset-bank001/BANK32-BALANCED-PROSPECTIVE002.json'),'sha256':a.BANK_SHA},
        bank_adapter={'path':str(root/'reset-bank001/reset_bank.py'),'sha256':a.BANK_ADAPTER_SHA},
        evaluation_seed=a.EVALUATION_SEED,max_output_bytes=a.LIMITS['raw_bytes'],deadline_epoch=intent['deadline_epoch'])


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


def selected_configuration(record, pilot_utils):
    """Frozen training factory only. Registration is selected by actual family."""
    validate_record(record)
    modules = {'timing_noise':'scripts.z1_timing_noise_training020','critic_noise':'scripts.z1_critic_noise_training021',
               'split_credit':'scripts.z1_split_credit_training021'}
    factory = importlib.import_module(modules[record['training_family']])
    task, config = factory.build_train_config(record['training_cell']['index'])
    need(task == record['task'] and factory.cell_for_index(record['training_cell']['index']) == record['training_cell'], 'selected frozen factory/task differs')
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from src.tasks.hammer.rl.runner import HammerOnPolicyRunner
    cell = record['training_cell']
    if record['training_family'] == 'split_credit':
        from src.tasks.hammer.config.z1.spline_split_credit_cfg import split_env_cfg, split_rl_cfg
        expected_env = split_env_cfg(play=False,setting=cell['setting']); expected_rl = split_rl_cfg(setting=cell['setting'])
    else:
        from src.tasks.hammer.config.z1.spline_exploration_pilot_cfg import pilot_env_cfg, pilot_rl_cfg
        expected_env = pilot_env_cfg(play=False,setting=cell['setting'],credit_deadline_ms=cell['credit_deadline_ms'],critic_mode=cell['critic_mode'])
        expected_rl = pilot_rl_cfg(setting=cell['setting'],critic_mode=cell['critic_mode'])
    need(pilot_utils.registered_screen_env_equal(load_env_cfg(task,play=False),expected_env)
         and load_rl_cfg(task) == expected_rl and load_runner_cls(task) is HammerOnPolicyRunner, 'selected registry/factory differs')
    need(config.env.scene.num_envs == 1024 and config.env.seed == cell['seed'] and config.agent.seed == cell['seed']
         and config.agent.max_iterations == 500 and config.agent.resume is False
         and config.sampled_pd500 is True and config.sampled_pd500_gain_mode == 'variable', 'original training factory semantics differ')
    # _condition performs the only measured overlay, world32 and evaluationseed.
    called = []
    def selected(task_name):
        need(task_name == task and not called, 'one selected training configuration call required')
        called.append(True)
        return copy.deepcopy(config.env),copy.deepcopy(config.agent)
    return selected


def execute_claimed_intent(intent):
    """One actual32-bank rollout. Requires T2 live ownership/timer/cap supervisor."""
    producer, _ = verified_inputs(intent)
    output = Path(intent['output']); private = Path(intent['private'])
    need(output.is_dir() and private.is_dir(), 'claimed fresh output/private directories required')
    from scripts.z1_timing_noise020 import compiler
    compiler.verify_receipt(producer,f"eval-{intent['index']:03d}",output,private)
    compiler.verify_configured(producer,private)
    need(not any(n == 'src' or n.startswith('src.') for n in sys.modules), 'scientific imports before physical adapter refused')
    pilot = pilot_utilities(); pins = verify_copies(); q = VENDOR/'pilot-compat001/qualified007'
    context = measurement_context(intent,producer)
    bank_path = Path(context['bank_adapter']['path'])
    need(sha_file(bank_path) == a.BANK_ADAPTER_SHA and sha_file(context['reset_bank']['path']) == a.BANK_SHA, 'bank bytes differ')
    origin = read_json(q/'current-evaluator-origin.json')
    need(sha_file(Path(producer['source_root'])/'scripts/evaluate_impact_speed.py') == origin['sha256'], 'qualified evaluator scientific origin differs')
    sys.path[:0] = [str(q),producer['source_root'],str(bank_path.parent)]
    import torch
    # Pinned R2 runtime-role check occurs before checkpoint deserialization.
    runtime_guard = importlib.import_module('scripts.z1_credit_critic_campaign')
    for module_name in runtime_guard.RUNTIME_MODULES.values():
        importlib.import_module(module_name)
    runtime_guard.verify_import_origins(producer,Path(producer['source_root']))
    need(not torch.cuda.is_available(), 'CPU evaluator must have CUDA hidden')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    payload = torch.load(intent['record']['checkpoint_path'],map_location='cpu',weights_only=False)
    write_once(output/'checkpoint-audit.json',validate_payload(payload,intent['record']))
    del payload
    import mjlab.tasks
    import src.tasks
    config_factory = selected_configuration(intent['record'],pilot)
    verify_import_origins(producer)
    qualified = load_pinned(q/'horizon_integration.py',pins['pilot-compat001/qualified007/horizon_integration.py'],'thirty_qualified_horizon023')
    evaluator = load_pinned(q/'current_evaluator_candidate.py',pins['pilot-compat001/qualified007/current_evaluator_candidate.py'],'thirty_qualified_evaluator023')
    bank = load_pinned(bank_path,a.BANK_ADAPTER_SHA,'thirty_bound_bank023')
    pilot.adapt_pilot_condition(evaluator,q/'current_evaluator_candidate.py')
    adapt_family_configuration(evaluator,q/'current_evaluator_candidate.py',config_factory)
    case = output/'pilot-bank32'
    args = evaluator.parse_args(evaluator_arguments(intent['record'],output=str(case),seed=a.EVALUATION_SEED))
    case.mkdir(exist_ok=False)
    adapter = make_adapter_class(qualified.HorizonAdapter,pilot)(context,case,bank)
    args.horizon_adapter = adapter
    # Time is rechecked immediately before construction/physical path. T2 also
    # enforces elapsed wall clock and owns interrupt/cleanup/caps throughout.
    validate_execution_intent(intent)
    verify_ref(intent['binding']['checkpoint'])
    runtime_guard.verify_import_origins(producer,Path(producer['source_root']))
    started_epoch = time.time()
    rows, report = evaluator._condition(args,None,0)
    finished_epoch = time.time()
    validate_execution_intent(intent,now=finished_epoch)
    need(adapter.started_counts == {world:1 for world in range(32)} and adapter.final_contract_restored is True,
         'exact first-start denominator/final restored contract differs')
    for item in [*rows,report]:
        item.update(observer_mapping=adapter.mapping_binding,native_onset_trace=adapter.trace_binding,
                    safety_endpoint_summary=adapter.endpoint_binding,reset_bank=context['reset_bank'])
    qualified.durable(case/'episodes.json',rows); qualified.durable(case/'report.json',report)
    verify_ref(intent['binding']['checkpoint']); verify_copies(); verify_import_origins(producer)
    compiler.verify_configured(producer,private)
    runtime_guard.verify_import_origins(producer,Path(producer['source_root']))
    return dict(status='RAW_REVIEW_REQUIRED',attempts=32,
        reports=[dict(case_id='pilot-bank32',report_sha256=sha_file(case/'report.json'),observer_mapping=adapter.mapping_binding,
                      native_onset_trace=adapter.trace_binding,safety_endpoint_summary=adapter.endpoint_binding)],
        observer_mappings=[adapter.mapping_binding],native_onset_traces=[adapter.trace_binding],safety_endpoint_summaries=[adapter.endpoint_binding],
        checkpoint_contract_validated_before_overlay=True,final_contract_restored=True,reset_frames_sampled=0,
        scientific_qualification=False,reset_bank=context['reset_bank'],policy_id=intent['record']['policy_id'],
        execution_timing=dict(started_epoch=started_epoch,finished_epoch=finished_epoch),
        configuration_adaptation_counts=evaluator.thirty_configuration_adaptation_counts,
        route_adaptation_counts=evaluator.pilot_condition_adaptation_counts)
