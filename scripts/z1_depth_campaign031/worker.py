"""Bounded031 children and six-slot/twenty-four-case finite controller."""
import contextlib
import hashlib
import importlib
import math
import os
from pathlib import Path
import re
import runpy
import selectors
import signal
import subprocess
import sys
import time
import uuid
from . import lifecycle as life
from .files import canonical,digest,inventory,need,read_json,reference,sha_file,verify_ref,write_once
from .manifest import LIMITS,ROOT,EVAL_ROOT,training_cell,validate,verify_package


def verify_import_origins(manifest):
    source=Path(manifest['source_root'])
    runtime=manifest['runtime_files']
    roles=('mjlab','mujoco','mujoco_warp','rsl_rl','torch')
    for name,module in tuple(sys.modules.items()):
        origin=getattr(module,'__file__',None)
        if name=='scripts' and origin is None:
            need(set(map(str,module.__path__))=={str(source/'scripts')},'ambient scripts namespace')
        elif name=='src' or name.startswith(('src.','scripts.')):
            if origin is None:continue
            path=Path(origin);need(path.is_relative_to(source) and sha_file(path)==manifest['source_files'].get(str(path.relative_to(source))), 'scientific origin escaped freeze: '+name)
        elif any(name==role or name.startswith(role+'.') for role in roles) and origin:
            path=str(Path(origin))
            if path in runtime:need(sha_file(path)==runtime[path],'runtime origin bytes differ: '+name)
            else:
                # The curated375 closure does not claim every native extension.
                site=str(Path(manifest['python']['path']).parent.parent/'lib/python3.12/site-packages')
                need(path.startswith(site+'/'),'runtime origin escaped pinned install: '+name)


def child_command(manifest,intent,sha):
    return [manifest['python']['path'],'-B','-I','-S',manifest['source_root']+'/scripts/z1_depth_campaign031/__main__.py',
        '--manifest',ROOT+'/manifest.json','--manifest-sha',digest(manifest),'child','--intent',str(intent),'--intent-sha',sha]


def _paths(manifest,kind,index):
    if kind=='training':
        need(type(index) is int and 0<=index<12,'cell outside finite scope');d=manifest['cells'][index];name=f'cell-{index:02d}'
    elif kind=='evaluation':
        need(type(index) is int and 0<=index<24,'case outside finite scope');d=manifest['cases'][index];name=f'case-{index:02d}'
    else:
        need(kind=='qualification' and type(index) is int and 0<=index<4,'gate outside finite scope');name=f'gate-{index}'
        d=dict(output=ROOT+'/qualification/'+name,private=ROOT+'/qualification/'+name+'-private')
    root=Path(EVAL_ROOT if kind=='evaluation' else ROOT)
    return name,Path(d['output']),Path(d['private']),root/'claims'/f'{name}.json',root/'intents'/f'{name}.json'


def _output_cap(kind):
    return LIMITS['training_output'] if kind=='training' else LIMITS['eval_output'] if kind=='evaluation' else 64*1024**2


def _private_cap(kind):
    return LIMITS['eval_private'] if kind=='evaluation' else LIMITS['training_private']


def _check_tree_caps(output,private,kind):
    need(life.tree_bytes(output)<=_output_cap(kind),'output tree cap exceeded')
    need(life.tree_bytes(private)<=_private_cap(kind),'private tree cap exceeded')
    if kind=='training':
        parts={'training-trace.jsonl':'trace','velocity-population.jsonl':'population'}
        for name,key in parts.items():
            path=output/name
            if path.exists():need(path.stat().st_size<=LIMITS[key],key+' cap exceeded')
        train=output/'train';checkpoints=sum(p.stat().st_size for p in train.glob('model_*.pt')) if train.exists() else 0
        need(checkpoints<=LIMITS['checkpoints'],'checkpoint aggregate cap exceeded')
        logs=sum((output/name).stat().st_size for name in ('stdout.log','stderr.log') if (output/name).exists())
        need(logs<=LIMITS['logs'],'combined logs cap exceeded')
        named=sum((output/name).stat().st_size for name in parts if (output/name).exists())
        need(life.tree_bytes(output)-checkpoints-logs-named<=LIMITS['other'],'other training output cap exceeded')


def run_one(manifest,grant,kind,index,allocation,deadline,*,binding=None):
    """Exclusive consumption; all errors after claim stay consumed, no retries."""
    deadline=min(deadline,grant['expires_epoch'])
    now=time.time();scope='qualification' if kind=='qualification' else 'campaign'
    life.validate_grant(manifest,grant,now=now,scope=scope);life.current_quota()
    seconds=manifest['qualification_recipe']['gates'][index]['seconds'] if kind=='qualification' else 7200
    need(now+seconds+60<=deadline,'HOLD_UNCLAIMED: case plus finalization cannot fit')
    name,output,private,claim,path=_paths(manifest,kind,index)
    need(not output.exists() and not private.exists() and not claim.exists() and not path.exists(),'fresh outputs/claim/intent required')
    if kind=='training' and index%4:
        from .evaluation import initial_anchor
        initial_anchor(manifest,index//4*4)
    output.mkdir();env=life.private_environment(private,cpu=kind=='evaluation')
    parent=life.process_identity(os.getpid());parent['argv']=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().split(b'\0')[:-1]
    parent['argv']=[b.decode() for b in parent['argv']]
    consumed=dict(schema='z1_depth_claim031',manifest_sha256=digest(manifest),grant_sha256=digest(grant),
        kind=kind,index=index,nonce=uuid.uuid4().hex,parent=parent,allocation=allocation,issued_epoch=now)
    write_once(claim,consumed)
    intent=dict(schema='z1_depth_intent031',manifest=manifest,grant=grant,kind=kind,index=index,
        output=str(output),private=str(private),parent=parent,allocation=allocation,issued_epoch=now,
        deadline_epoch=min(now+seconds,deadline-60),claim=reference(claim),nonce=consumed['nonce'])
    if binding is not None:
        from .evaluation import record_for_binding
        intent['binding']=binding;intent['record']=record_for_binding(manifest,index,binding)
    write_once(path,intent)
    runner=life._Runner();child=None;selector=selectors.DefaultSelector();logs={};total=0
    started=time.monotonic();next_check=started;error=None
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def interrupted(sig,frame):raise RuntimeError('owned allocation interrupted: '+str(sig))
    for sig in previous:signal.signal(sig,interrupted)
    try:
        for key in ('stdout','stderr'):logs[key]=(output/(key+'.log')).open('xb')
        # Block signals across spawn/handle installation, then restore immediately.
        oldmask=signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
        try:
            child=runner.start(child_command(manifest,path,sha_file(path)),env,subprocess.PIPE,subprocess.PIPE)
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)
        for key in ('stdout','stderr'):
            stream=getattr(child.p,key);os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ,key)
        while selector.get_map() or child.poll() is None:
            now=time.time()
            need(now<intent['deadline_epoch'] and time.monotonic()-started<seconds,'child hard deadline exceeded')
            for key,_ in selector.select(.2):
                data=os.read(key.fileobj.fileno(),65536)
                if not data:selector.unregister(key.fileobj);key.fileobj.close();continue
                total+=len(data);need(total<=LIMITS['logs'],'combined child logs cap exceeded');logs[key.data].write(data)
            if time.monotonic()>=next_check:
                _check_tree_caps(output,private,kind);life.current_quota();next_check=time.monotonic()+3
                if kind=='training':
                    from .evaluation import publish_seals
                    publish_seals(manifest,index)
        need(child.wait()==0,'child exited nonzero')
        life._cleanup(child);child=None
        _check_tree_caps(output,private,kind)
        if kind!='evaluation':
            gpu=read_json(output/'gpu-runtime.json');need(gpu==dict(logical_device=0,gpu_uuid=allocation['gpu_uuid']),'actual child GPU UUID differs')
        result=read_json(output/'child-result.json')
        need(result.get('intent_sha256')==sha_file(path) and result.get('manifest_sha256')==digest(manifest),'child result lineage differs')
        need(time.time()<deadline,'finalization deadline exceeded')
        write_once(output/'terminal.json',dict(schema='z1_depth_terminal031',status='COMPLETE',index=index,kind=kind,
            manifest_sha256=digest(manifest),claim=reference(claim),result=reference(output/'child-result.json')))
        return result
    except BaseException as exc:
        error=exc
        if child is not None:
            try:life._cleanup(child)
            except BaseException as cleanup:error=RuntimeError(str(exc)+'; cleanup UNKNOWN: '+str(cleanup))
        if not (output/'terminal.json').exists():
            write_once(output/'terminal.json',dict(schema='z1_depth_terminal031',status='FAILED_NO_RETRY',index=index,kind=kind,
                manifest_sha256=digest(manifest),claim=reference(claim),error=str(error),scientific_outcome='UNKNOWN'))
        raise error
    finally:
        selector.close()
        for stream in logs.values():stream.flush();os.fsync(stream.fileno());stream.close()
        for sig,handler in previous.items():signal.signal(sig,handler)


def _validate_child(intent,path,sha):
    need(sha_file(path)==sha and intent['schema']=='z1_depth_intent031','child intent SHA/schema differs')
    manifest=intent['manifest'];kind=intent['kind'];index=intent['index']
    life.validate_grant(manifest,intent['grant'],now=time.time(),scope='qualification' if kind=='qualification' else 'campaign')
    name,output,private,claim,expected=_paths(manifest,kind,index)
    need(str(path)==str(expected) and str(output)==intent['output'] and str(private)==intent['private'],'child paths differ')
    verify_ref(intent['claim'],claim);consumed=read_json(claim)
    need(consumed['nonce']==intent['nonce'] and consumed['manifest_sha256']==digest(manifest) and consumed['parent']==intent['parent'],'claim lineage differs')
    parent=life.process_identity(os.getppid())
    need(parent=={k:intent['parent'][k] for k in ('pid','startticks')},'living owned parent differs')
    parent_argv=[b.decode() for b in Path(f"/proc/{parent['pid']}/cmdline").read_bytes().split(b'\0')[:-1]]
    need(parent_argv==intent['parent']['argv'],'living parent command differs')
    actual=[b.decode() for b in Path(f'/proc/{os.getpid()}/cmdline').read_bytes().split(b'\0')[:-1]]
    need(actual==child_command(manifest,path,sha),'exact isolated child argv differs')
    need(time.time()<intent['deadline_epoch']<=intent['allocation']['end_epoch'],'expired child')
    need(os.environ.get('SLURM_JOB_ID')==intent['allocation']['job_id'],'allocation binding differs')
    import ctypes
    need(ctypes.CDLL(None).prctl(1,signal.SIGTERM)==0 and os.getppid()==parent['pid'],'parent death binding unavailable')
    return manifest


def run_canonical(manifest,gate,output):
    """Canonical argv and bodies unchanged; observe native start/reset counts."""
    from mjlab.envs import ManagerBasedRlEnv
    counts=dict(constructor_worlds=0,reset_worlds=0,world_transitions=0,native_autoreset_worlds=0,learned_forwards=0,updates=0)
    active={'step':False};original={key:getattr(ManagerBasedRlEnv,key) for key in ('__init__','reset','step','_reset_idx')}
    def init(env,*args,**kwargs):
        cfg=kwargs.get('cfg',args[0] if args else None)
        counts['constructor_worlds']+=cfg.scene.num_envs
        original['__init__'](env,*args,**kwargs)
    def reset(env,*args,**kwargs):
        counts['reset_worlds']+=env.num_envs;return original['reset'](env,*args,**kwargs)
    def reset_idx(env,ids,*args,**kwargs):
        if active['step']:counts['native_autoreset_worlds']+=len(ids)
        return original['_reset_idx'](env,ids,*args,**kwargs)
    def step(env,*args,**kwargs):
        counts['world_transitions']+=env.num_envs
        need(counts['world_transitions']<=gate['world_transitions'],'canonical transition cap exceeded')
        active['step']=True
        try:return original['step'](env,*args,**kwargs)
        finally:active['step']=False
    for name,fn in dict(__init__=init,reset=reset,step=step,_reset_idx=reset_idx).items():setattr(ManagerBasedRlEnv,name,fn)
    try:
        script=Path(manifest['source_root'])/gate['script'];sys.argv=[str(script)]
        scope=runpy.run_path(str(script),run_name='depth031_canonical_gate');verify_import_origins(manifest);scope['main']()
        need(counts['constructor_worlds']==gate['constructor_worlds'] and counts['reset_worlds']==gate['reset_worlds'],'canonical actual start/reset count differs')
        return dict(status='GATE_EXIT_ZERO',counts=counts)
    finally:
        for name,fn in original.items():setattr(ManagerBasedRlEnv,name,fn)
        write_once(Path(output)/'actual-start-counts.json',counts)


def audit_training(manifest,index,output):
    from .evaluation import checkpoint_seal
    from src.tasks.hammer.rl.depth_reward031_trace import validate_depth_trace_row031
    previous=None;rows=0
    with (output/'training-trace.jsonl').open() as stream:
        import json
        for row in stream:
            need(len(row)<=512*1024,'trace row cap exceeded')
            previous=validate_depth_trace_row031(json.loads(row),rows,previous=previous);rows+=1
            need(rows<=500,'extra training updates')
    need(rows==500,'exact500 trace updates required')
    seals=[checkpoint_seal(manifest,index,label) for label in (0,24,49,99,249,499)]
    need(read_json(output/'checkpoint-499-final-save-deduplicated.json')['deduplicated_final_save'] is True,'final deduplication receipt required')
    with (output/'velocity-population.jsonl').open() as stream:
        population_rows=[json.loads(line) for line in stream]
    validate_population(population_rows,threshold=3.1415)
    population=len(population_rows)
    return dict(schema='z1_depth_training_audit031',status='COMPLETE',index=index,manifest_sha256=digest(manifest),
        checkpoints=seals,trace_rows=rows,population_rows=population,raw_files={name:ref for name,ref in inventory(output,LIMITS['training_output']).items() if name not in ('stdout.log','stderr.log')})


def execute_child(path,sha):
    intent=read_json(path);manifest=_validate_child(intent,path,sha)
    claim_path=Path(intent['claim']['path'])
    write_once(claim_path.with_suffix('.child.json'),dict(schema='z1_depth_child_consumed031',identity=life.process_identity(os.getpid()),
        parent=intent['parent'],intent_sha256=sha,manifest_sha256=digest(manifest),nonce=intent['nonce']))
    verify_package(manifest)
    private=Path(intent['private']);output=Path(intent['output']);os.chdir(private)
    site=Path(manifest['python']['path']).parent.parent/'lib/python3.12/site-packages';sys.path.append(str(site))
    from . import compiler
    compiler.configure(manifest,private,output,f"{intent['kind']}-{intent['index']}")
    kind=intent['kind'];index=intent['index'];started=time.time()
    if kind!='evaluation':
        import torch
        write_once(output/'gpu-runtime.json',life.verify_cuda_device(torch,intent['allocation']))
        torch.set_num_threads(4);torch.set_num_interop_threads(4)
    try:
        if kind=='qualification':
            gate=manifest['qualification_recipe']['gates'][index]
            if index<3:result=run_canonical(manifest,gate,output)
            else:
                from .probe import execute_probe
                result=execute_probe(output,manifest['qualification_recipe'])
        elif kind=='training':
            from scripts.z1_depth_reward_training031 import build_train_config,matched_initial_state_gate,selected_checkpoint_saves
            from scripts.train import run_train
            from .evaluation import initial_anchor,publish_seals
            task,cfg=build_train_config(index);cfg.env.sim.nan_guard.output_dir=str(private/'nan-dumps');ref=None if index%4==0 else initial_anchor(manifest,index//4*4)
            verify_import_origins(manifest)
            with selected_checkpoint_saves(index,output),matched_initial_state_gate(index,output,reference_initial_state=ref):
                run_train(task,cfg,output/'train')
            publish_seals(manifest,index);result=audit_training(manifest,index,output)
            write_once(output/'audit.json',result)
        else:
            from .evaluation import execute,audit_case
            execute(intent);result=audit_case(intent);write_once(output/'audit.json',result)
        compiler.verify_configured(manifest,private);verify_import_origins(manifest)
        need(time.time()<intent['deadline_epoch'],'child completion exceeded deadline')
        result.update(manifest_sha256=digest(manifest),intent_sha256=sha,started_epoch=started,finished_epoch=time.time())
        write_once(output/'child-result.json',result)
        return result
    except BaseException as exc:
        write_once(output/'failure.json',dict(status='FAILED_NO_RETRY',error=str(exc),manifest_sha256=digest(manifest)))
        raise


def qualify(manifest,grant):
    verify_package(manifest)
    alloc=life.allocation(os.environ['SLURM_JOB_ID'],'depth031-qualification')
    life.validate_grant(manifest,grant,now=time.time(),scope='qualification');life.current_quota()
    clock=life.start_clock(Path(ROOT)/'clock.json',digest(manifest),alloc['job_id'],alloc['start_epoch'])
    deadline=life.admit_time(clock,now=time.time(),allocation_end=alloc['end_epoch'],bound=2470)
    results=[]
    for index,gate in enumerate(manifest['qualification_recipe']['gates']):
        # Admission rechecks before each irreversible gate claim.
        need(time.time()+sum(g['seconds']+70 for g in manifest['qualification_recipe']['gates'][index:])+270<=deadline,
             'remaining canonical/probe/finalization bounds cannot fit')
        result=run_one(manifest,grant,'qualification',index,alloc,deadline)
        if index<3:
            log=Path(ROOT)/'qualification'/f'gate-{index}'/'stdout.log'
            need(gate['pass_marker'] in log.read_text(),'canonical PASS marker absent')
        else:need(result.get('status')=='PROBE_ASSERTIONS_PASSED','031 probe assertions incomplete')
        results.append(reference(Path(ROOT)/'qualification'/f'gate-{index}'/'child-result.json'))
    receipt=dict(schema='z1_depth_qualification031',status='PASS',manifest_sha256=digest(manifest),
        grant_sha256=digest(grant),recipe_sha256=digest(manifest['qualification_recipe']),allocation=alloc,
        clock=reference(Path(ROOT)/'clock.json'),gates=results,scientific_qualification=False)
    need(time.time()+270<=deadline,'qualification finalization reserve exhausted')
    write_once(Path(ROOT)/'qualification-receipt.json',receipt)
    return receipt


def verify_qualification(manifest):
    receipt=read_json(Path(ROOT)/'qualification-receipt.json')
    need(receipt.get('schema')=='z1_depth_qualification031' and receipt.get('status')=='PASS'
         and receipt.get('manifest_sha256')==digest(manifest) and receipt.get('recipe_sha256')==digest(manifest['qualification_recipe'])
         and len(receipt.get('gates',[]))==4,'new complete031 qualification required')
    verify_ref(receipt['clock'],Path(ROOT)/'clock.json')
    for ref in receipt['gates']:verify_ref(ref)
    row=life.slurm_job(receipt['allocation']['job_id'])
    need(life.qualification_terminal(row),'qualification allocation must terminate COMPLETED/0 before GPU training submissions')
    need(row.get('JobName')=='depth031-qualification' and row.get('UserId','').split('(')[0]=='eunikhilr','qualification owner differs')
    return receipt


def run_slot(manifest,grant,slot):
    need(type(slot) is int and 0<=slot<6,'exact slot0..5 required')
    verify_package(manifest);verify_qualification(manifest)
    alloc=life.allocation(os.environ['SLURM_JOB_ID'],f'depth031-slot-{slot}')
    clock=read_json(Path(ROOT)/'clock.json')
    need(clock['manifest_sha256']==digest(manifest),'global clock belongs to different release')
    need(alloc['deadline_epoch']==clock['deadline_epoch']-180 and alloc['time_limit_seconds']==16200,'scheduler deadline/slot timelimit differs')
    deadline=life.admit_time(clock,now=time.time(),allocation_end=alloc['end_epoch'],bound=16030)
    life.validate_grant(manifest,grant,now=time.time(),scope='campaign');life.current_quota()
    from .evaluation import initial_anchor
    if slot%2:
        wait_until=min(time.time()+600,deadline-15430)
        while not (Path(manifest['cells'][4*(slot//2)]['output'])/'initial-seal.json').exists():
            need(time.time()<wait_until,'HOLD_UNCLAIMED: bounded600s A0 anchor wait expired');time.sleep(1)
        initial_anchor(manifest,4*(slot//2))
    for index in manifest['slots'][slot]['cells']:
        run_one(manifest,grant,'training',index,alloc,deadline)


def run_evaluation(manifest,grant,index):
    need(type(index) is int and 0<=index<24,'exact evaluation case0..23 required')
    verify_package(manifest);verify_qualification(manifest)
    alloc=life.allocation(os.environ['SLURM_JOB_ID'],f'depth031-eval-{index}')
    clock=read_json(Path(ROOT)/'clock.json')
    deadline=life.admit_evaluation_time(manifest,clock,alloc,grant,now=time.time())
    d=manifest['cases'][index]
    binding=read_json(Path(manifest['cells'][d['cell']]['output'])/f"model-{d['label']}-seal.json")
    run_one(manifest,grant,'evaluation',index,alloc,deadline,binding=binding)


def submission_plan(manifest,clock=None):
    """Finite closed commands. Root reviews this before granting campaign scope."""
    from datetime import datetime,timezone
    if clock is None and (Path(ROOT)/'clock.json').is_file():clock=read_json(Path(ROOT)/'clock.json')
    if clock is not None:
        need(clock['manifest_sha256']==digest(manifest) and clock['deadline_epoch']==clock['first_allocation_epoch']+28800,'submission global clock differs')
    cpu_deadline='CPU_UTC_DEADLINE_FROM_CLOCK' if clock is None else datetime.fromtimestamp(clock['first_allocation_epoch']+64800-180,timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    gpu_deadline='GLOBAL_UTC_DEADLINE_FROM_CLOCK' if clock is None else datetime.fromtimestamp(clock['deadline_epoch']-180,timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    sha=digest(manifest);entry=manifest['source_root']+'/scripts/z1_depth_campaign031/__main__.py'
    python=manifest['python']['path'];common=[python,'-B','-I','-S',entry,'--manifest',ROOT+'/manifest.json','--manifest-sha',sha]
    jobs=[]
    for kind,count in (('training',6),('evaluation',24)):
        for index in range(count):
            cpu=kind=='evaluation';name=f'depth031-{"eval" if cpu else "slot"}-{index}'
            command=common+['evaluation' if cpu else 'slot','--grant',ROOT+'/campaign-grant.json','--index',str(index)]
            # Separate cases are throttled globally by the finite controller to7.
            sbatch=['sbatch','--parsable','--no-requeue','--export=NONE','--job-name='+name,'--comment=depth031:'+sha,
                '--account='+manifest['slurm']['account'],'--qos='+manifest['slurm']['qos'],
                '--partition='+('cpu' if cpu else 'gpu'),'--cpus-per-task='+('2' if cpu else '4'),'--mem='+('8G' if cpu else '32G'),
                '--time='+('02:15:00' if cpu else '04:30:00'),'--output='+ROOT+f'/slurm/{name}-%j.log']
            if not cpu:sbatch+=['--gres=gpu:1','--deadline='+gpu_deadline]
            else:sbatch+=['--deadline='+cpu_deadline]
            import shlex
            sbatch+=['--wrap','exec '+shlex.join(command)]
            jobs.append(dict(kind=kind,index=index,name=name,command=sbatch))
    return dict(schema='z1_depth_submission031',manifest_sha256=sha,jobs=jobs,training_slots=6,evaluation_cases=24,
        gpu_concurrency=6,eval_concurrency=7,controller_seconds=64800,controller_cpus=1,controller_memory_gib=2,
        gpu_deadline_seconds=28800,retry=False,requeue=False,initial_slots=[0,2,4],paired_slots_require_initial_anchor=True,
        qualification_requires_terminal_completed=True,slurm_gpu_deadline_utc=gpu_deadline,slurm_cpu_case_deadline_utc=cpu_deadline,scheduler_exit_margin_seconds=180,case_cpu_hours_bound=108,controller_cpu_hours_bound=18)


def _submit_once(manifest,plan,kind,index):
    need(kind in ('training','evaluation'),'closed finite submission kind required')
    candidates=[j for j in plan['jobs'] if j['kind']==kind and j['index']==index]
    need(len(candidates)==1,'job not in reviewed allowlist')
    job=candidates[0];path=Path(ROOT)/'submissions'/f'{kind}-{index:02d}.json'
    # Persist consumption before sbatch: ambiguous submissions are NEVER retried.
    pending=path.with_suffix('.consumed.json')
    write_once(pending,dict(schema='z1_depth_submission_consumed031',manifest_sha256=digest(manifest),job=job,epoch=time.time()))
    result=subprocess.run(job['command'],check=True,capture_output=True,text=True,timeout=30,env=dict(os.environ,TZ='UTC'))
    job_id=result.stdout.strip().split(';')[0]
    need(bool(re.fullmatch('[1-9][0-9]*',job_id)),'ambiguous sbatch output; submission consumed, no retry')
    row=dict(schema='z1_depth_submitted031',manifest_sha256=digest(manifest),kind=kind,index=index,
        job_id=job_id,name=job['name'],submitted_epoch=time.time(),consumption=reference(pending))
    write_once(path,row);return row


def controller(manifest,grant):
    verify_package(manifest);verify_qualification(manifest)
    alloc=life.allocation(os.environ['SLURM_JOB_ID'],'depth031-controller')
    life.validate_grant(manifest,grant,now=time.time(),scope='campaign')
    plan=submission_plan(manifest);saved=read_json(Path(ROOT)/'submission-manifest.json')
    need(saved==plan,'independently reviewed submission manifest differs')
    need(any(ref['path']==ROOT+'/submission-manifest.json' and ref['sha256']==sha_file(ROOT+'/submission-manifest.json') for ref in grant['review_refs']),
         'grant must bind reviewed exact submission manifest')
    write_once(Path(ROOT)/'controller-claim.json',dict(schema='z1_depth_controller031',manifest_sha256=digest(manifest),allocation=alloc))
    clock=read_json(Path(ROOT)/'clock.json');entries=[];submitted_slots=set();submitted_cases=set()
    from .evaluation import initial_anchor,publish_seals
    deadline=min(alloc['end_epoch']-120,clock['first_allocation_epoch']+64800-180,grant['expires_epoch']-120)
    try:
        while time.time()<deadline:
            now=time.time();life.current_quota()
            for index in range(12):publish_seals(manifest,index)
            anchors=set()
            for index in (0,4,8):
                if (Path(manifest['cells'][index]['output'])/'initial-seal.json').exists():initial_anchor(manifest,index);anchors.add(index)
            if now>=clock['deadline_epoch']:
                life.cancel_expired_gpu_jobs(entries,clock,now=now)
            else:
                for slot in life.ready_slots(submitted_slots,anchors):
                    if now+16200>clock['deadline_epoch']-180:continue
                    entries.append(_submit_once(manifest,plan,'training',slot));submitted_slots.add(slot)
            running_evals=0;terminal=0
            for entry in entries:
                row=life.slurm_job(entry['job_id']);life.validate_owned_job(entry,row,digest(manifest))
                active=row['JobState'] in ('RUNNING','PENDING','CONFIGURING','COMPLETING')
                if entry['kind']=='evaluation' and active:running_evals+=1
                if not active:terminal+=1
            for index,d in enumerate(manifest['cases']):
                if index in submitted_cases or running_evals>=7:continue
                seal=Path(manifest['cells'][d['cell']]['output'])/f"model-{d['label']}-seal.json"
                if seal.is_file() and now+8100+120<=deadline:
                    entries.append(_submit_once(manifest,plan,'evaluation',index));submitted_cases.add(index);running_evals+=1
            if terminal==len(entries) and (len(submitted_cases)==24 or now>=clock['deadline_epoch']):break
            time.sleep(10)
    finally:
        # Controller never cancels unrelated jobs. GPU deadline enforcement persists
        # for the full finite controller lifetime; worker wallcaps independently bound.
        if time.time()>=clock['deadline_epoch']:life.cancel_expired_gpu_jobs(entries,clock,now=time.time())
        uncertain=[]
        for pending in (Path(ROOT)/'submissions').glob('*.consumed.json'):
            returned=pending.with_name(pending.name.replace('.consumed.json','.json'))
            if not returned.exists():uncertain.append(reference(pending))
        write_once(Path(ROOT)/'controller-final.json',dict(schema='z1_depth_controller_final031',manifest_sha256=digest(manifest),
            submissions=entries,submission_uncertainties=uncertain,absolute_cpu_application_cutoff=deadline,unsubmitted_slots=sorted(set(range(6))-submitted_slots),unsubmitted_cases=sorted(set(range(24))-submitted_cases),
            status='FINITE_CONTROLLER_EXIT_NOT_SCIENTIFIC_SUCCESS'))


# Narrow copied audited population/censoring arithmetic; no old authority imports.
_EPISODE_KEYS=('all_episode_starts','observed_episode_starts','observed_contact_onsets',
    'native_contact_episodes','native_no_contact_episodes','native_completed_productive_episodes',
    'observed_productive_contact_episodes','external_reset_observed_productive_episodes',
    'native_timeout_episodes','contact_resets_before_200ms','contact_policy_envelopes_reached_200ms',
    'external_reset_episodes','external_contact_censored_episodes')

def _number(value,low=None,high=None,integer=False):
    need(type(value) in ((int,) if integer else (int,float)) and math.isfinite(value),
         'finite population numeric domain required')
    need((low is None or value>=low) and (high is None or value<=high),'population numeric bound differs')
    return value

def _episode_counts(row,previous=None):
    need(type(row) is dict and set(row)==set(_EPISODE_KEYS),'complete population episode counters required')
    for key in _EPISODE_KEYS:
        _number(row[key],0,1024+1024*12000,integer=True)
        if previous is not None:need(row[key]>=previous[key],'population cumulative counter decreased')
    ncontact,nmiss,external=(row[k] for k in ('native_contact_episodes','native_no_contact_episodes','external_reset_episodes'))
    need(row['all_episode_starts']==1024+ncontact+nmiss+external,'population episode-start conservation differs')
    need(row['contact_resets_before_200ms']+row['contact_policy_envelopes_reached_200ms']==ncontact,
         'population contact-window conservation differs')
    need(row['native_completed_productive_episodes']<=ncontact and row['native_timeout_episodes']<=ncontact+nmiss
         and row['external_contact_censored_episodes']<=external
         and row['external_reset_observed_productive_episodes']<=external,'population episode subcounts differ')
    need(row['observed_contact_onsets']<=row['observed_episode_starts']<=row['all_episode_starts']
         and row['observed_productive_contact_episodes']<=row['observed_contact_onsets'],
         'population observed-episode denominator differs')

def validate_population(rows,*,threshold):
    _number(threshold,0);need(threshold>0,'positive live population threshold required')
    need(type(rows) is list and len(rows)==501,'500 population updates plus final row required')
    previous=None
    vector_keys=('peak_rad_s_max_per_joint','peak_rad_s_mean_per_joint','raw_margin_max_per_joint',
        'exceedance_policy_steps_per_joint','exceedance_policy_step_fraction_per_joint',
        'probability_mean_per_joint','probability_max_per_joint','ema_min_per_joint','ema_max_per_joint')
    for update,row in enumerate(rows[:-1]):
        need(type(row) is dict and row.get('schema')=='z1_velocity_population_v1' and row.get('kind')=='update'
             and type(row.get('update')) is int and row['update']==update,'population update order/schema differs')
        need(row.get('scope')=='ALL_WORLDS_POLICY_STEP_ENVELOPE'
             and type(row.get('worlds')) is int and row['worlds']==1024
             and type(row.get('world_transitions')) is int and row['world_transitions']==24576
             and type(row.get('common_step_counter')) is int and row['common_step_counter']==24*(update+1),
             'population world/transition clock differs')
        need('exact_200ms_contact_window_safety' in row and row['exact_200ms_contact_window_safety'] is None
             and row.get('envelope_caveat')=='ONSET_STEP_MAY_INCLUDE_PRECONTACT_FINAL_STEP_MAY_EXCEED_200MS',
             'population must preserve unknown exact-window safety')
        velocity=row.get('velocity')
        need(type(velocity) is dict and set(velocity)==set(vector_keys)|{'threshold_rad_s','joint_names'},
             'complete population velocity schema required')
        _number(velocity['threshold_rad_s'],0)
        need(velocity['threshold_rad_s']==threshold
             and velocity['joint_names']==[f'joint{i}' for i in range(1,7)],'population threshold/joint identity differs')
        for key in vector_keys:
            values=velocity[key];need(type(values) is list and len(values)==6,'population six-joint vector required')
            for value in values:
                low=None if key=='raw_margin_max_per_joint' else 0
                high=1 if key in ('probability_mean_per_joint','probability_max_per_joint','exceedance_policy_step_fraction_per_joint') else (24576 if key=='exceedance_policy_steps_per_joint' else None)
                _number(value,low,high,integer=key=='exceedance_policy_steps_per_joint')
        for j in range(6):
            at=lambda key:velocity[key][j]
            need(at('peak_rad_s_mean_per_joint')<=at('peak_rad_s_max_per_joint')+1e-5
                 and at('raw_margin_max_per_joint')+threshold<=at('peak_rad_s_max_per_joint')+1e-4,
                 'population raw peak/margin domains differ')
            need(math.isclose(at('exceedance_policy_step_fraction_per_joint'),at('exceedance_policy_steps_per_joint')/24576,
                              rel_tol=1e-5,abs_tol=1e-7),'population exceedance fraction denominator differs')
            need(at('probability_mean_per_joint')<=at('probability_max_per_joint')+1e-6
                 and at('ema_min_per_joint')<=at('ema_max_per_joint'),'population probability/EMA domains differ')
        winners=row.get('pressure_winner_policy_steps')
        need(type(winners) is dict and set(winners)=={'none','velocity','impulse','tie'},'population pressure winners required')
        for value in winners.values():_number(value,0,24576,integer=True)
        need(sum(winners.values())==24576,'population pressure winner conservation differs')
        counts=row.get('episodes_cumulative');_episode_counts(counts,previous);previous=counts
        canonical(row)  # Reject nonfinite extra fields as well.
    final=rows[-1]
    need(type(final) is dict and final.get('schema')=='z1_velocity_population_v1' and final.get('kind')=='final'
         and final.get('scope')=='ALL_WORLDS_POLICY_STEP_ENVELOPE'
         and final.get('status')=='COMPLETE_OBSERVATION_NOT_SAFETY_CERTIFICATION'
         and type(final.get('updates')) is int and final['updates']==500
         and type(final.get('partial_policy_steps')) is int and final['partial_policy_steps']==0,
         'complete population final envelope required')
    need('exact_200ms_contact_window_safety' in final and final['exact_200ms_contact_window_safety'] is None,
         'final exact-window safety must remain unknown')
    counts={k:final.get(k) for k in _EPISODE_KEYS};_episode_counts(counts,previous)
    need(counts==previous,'final population counts changed after final update')
    for key in ('active_at_stop_right_censored','active_contact_at_stop','unobserved_episode_starts_at_stop','active_observed_productive_at_stop'):
        _number(final.get(key),0,1024,integer=True)
    need(final['active_at_stop_right_censored']+final['unobserved_episode_starts_at_stop']==1024
         and final['active_contact_at_stop']<=final['active_at_stop_right_censored']
         and final['active_observed_productive_at_stop']<=final['active_contact_at_stop'],
         'final population censoring conservation differs')
    need(counts['observed_productive_contact_episodes']==counts['native_completed_productive_episodes']
         +counts['external_reset_observed_productive_episodes']+final['active_observed_productive_at_stop'],
         'final productive population conservation differs')
    canonical(final)

