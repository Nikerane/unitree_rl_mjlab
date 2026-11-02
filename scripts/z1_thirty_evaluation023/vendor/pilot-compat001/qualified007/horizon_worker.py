"""Candidate worker admission shell; integration hook remains separately reviewed.

The hook must load the exact current task/checkpoint, validate its metadata BEFORE
installing the copied observer/continuation, and return durable first-start rows.
No hook or release is supplied, so this file cannot execute a live evaluation as
staged. It does not substitute historical validators or task configurations.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import types

from horizon_eval_preflight import TASKS, BANKS

TRUSTED_SUPERVISOR_SHA256 = "f4b40175786581063848c8b8d9b40651f203f9670418fcbe6d3da7274f3af449"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, data):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        json.dump(data, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def require_supervised(intent_path,intent):
    context=intent.get('supervisor_context')
    if not isinstance(context,dict) or context.get('parent')!=os.getppid():
        raise ValueError('physical worker requires its live bounded supervisor')
    supervisor=Path(context.get('supervisor_path','')).resolve(strict=True)
    trusted=(Path(__file__).resolve().parent/'supervisor'/'cpu_supervisor.py').resolve(strict=True)
    if supervisor!=trusted or context.get('supervisor_sha256')!=TRUSTED_SUPERVISOR_SHA256:
        raise ValueError('trusted supervisor path/context mismatch')
    source=supervisor.read_bytes()
    if hashlib.sha256(source).hexdigest()!=TRUSTED_SUPERVISOR_SHA256:
        raise ValueError('trusted supervisor bytes mismatch')
    module=types.ModuleType('trusted_cpu_supervisor_guard');module.__file__=str(supervisor)
    exec(compile(source,str(supervisor),'exec'),module.__dict__)  # Execute exactly the buffer whose SHA was checked; never cached bytecode.
    result=module.authorize_physical_worker(intent_path,Path(__file__).resolve())
    if result.get('status')!='AUTHORIZED_LIVE_SUPERVISOR':raise ValueError('supervisor guard did not admit this child')
    if intent['scope']=='science160':
        manifest=module.load_release(context['release_path'])
        reference=manifest['slots'][context['slot']]['qualification_seal']
        if reference!={'path':intent['own_cpu32_seal_path'],'sha256':intent['own_cpu32_seal_sha256']}:
            raise ValueError('own CPU32 seal reference differs from admitted slot')
        module.verify_qualification(intent,reference)  # Recompute from pinned source before the physical claim.
    return result


def run(intent_path):
    intent_path=Path(intent_path).resolve(strict=True)
    intent=json.loads(intent_path.read_text())
    if intent['schema']!='z1_horizon_cpu_worker_v1' or intent['status']!='ADMITTED_ONCE':
        raise ValueError('exact new worker admission required')
    if intent['retry_allowed'] is not False or intent['resume_allowed'] is not False:
        raise ValueError('no retry/resume')
    admission=require_supervised(intent_path,intent)
    policy=intent['policy'];arm=policy['arm'];scope=intent['scope']
    if policy['task']!=TASKS[arm] or policy['native_window_s']!={'native50':.05,'extended200':.2}[arm]:
        raise ValueError('task/arm/window mismatch')
    expected=32 if scope=='own_cpu32' else 160
    if scope not in ('own_cpu32','science160') or intent['attempts']!=expected:
        raise ValueError('bounded population required')
    if scope=='science160' and intent['evaluation_seed'] not in BANKS:
        raise ValueError('undeclared scientific bank')
    for group in ('source_files','asset_files','package_files','training_evidence'):
        if not intent[group]:raise ValueError('empty sealed inventory: '+group)
        for row in intent[group]:
            if digest(row['path'])!=row['sha256']:raise ValueError('sealed file differs')
    if digest(policy['checkpoint_path'])!=policy['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    root=Path(intent['output_root']).resolve(strict=True)
    if intent_path.parent!=root or time.time()>=intent['deadline_epoch']:
        raise ValueError('output/deadline binding differs')
    # Claims live in the cohort-wide ledger, never under movable output roots.
    claim=Path(intent['claim_path'])
    if claim.name!=intent['claim_key']+'.json':raise ValueError('claim binding differs')
    write_once(claim,{'intent_sha256':digest(intent_path),'policy':policy,'scope':scope,
                      'bank':intent['evaluation_seed'],'attempts':expected})
    hook=Path(intent['reviewed_hook_path']).resolve(strict=True)
    if digest(hook)!=intent['reviewed_hook_sha256']:raise ValueError('hook differs')
    spec=importlib.util.spec_from_file_location('sealed_horizon_hook',hook)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # The integration implementation must perform the source/runtime/contract
    # checks again at imports, each case, overlay install, and final restoration.
    result=module.execute_horizon_intent(intent)
    if (result['attempts']!=expected or result['checkpoint_contract_validated_before_overlay'] is not True
            or result['final_contract_restored'] is not True or result['reset_frames_sampled']!=0):
        raise ValueError('hook did not satisfy required evidence contract')
    result['hardware_prolog_sha256']=admission['hardware_prolog_sha256']
    write_once(root/'worker-result.json',result)
    return result
