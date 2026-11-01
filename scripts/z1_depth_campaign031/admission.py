"""Fixed corrected031 admission; all three failed attempts remain consumed."""
import hashlib
import math
import re
import stat
from pathlib import Path
from .files import canonical, digest, inventory, need, read_json, safe_path, verify_ref

FIRST_ROOTS=('/cephhome/eunikhilr/campaigns/z1-depth-reward-20261101-v1',
             '/cephhome/eunikhilr/campaigns/z1-depth-evaluation-20261101-v1')
SECOND_ROOTS=tuple(root[:-1]+'2' for root in FIRST_ROOTS)
THIRD_ROOTS=tuple(root[:-1]+'3' for root in FIRST_ROOTS)
PRESERVED_ROOTS=FIRST_ROOTS+SECOND_ROOTS+THIRD_ROOTS
SUCCESSOR_ROOTS=tuple(root[:-1]+'4' for root in FIRST_ROOTS)
SCOPE_ROOTS=PRESERVED_ROOTS+SUCCESSOR_ROOTS
ORIGIN_REF=dict(path=FIRST_ROOTS[0]+'/controls/first-gpu-allocation-accounting031.json',
    sha256='b45b58d58a7f8634201bc33b6b12dcd32900f867251a144d64755be12e543a20',size=1557)
SECOND_REF=dict(path=SECOND_ROOTS[0]+'/controls/second-technical-allocation-accounting031.json',
    sha256='4056c9b6a6132356934166d7724c350f2aa7034e0fd8eb1ff1ee1b3858899e29',size=5013)
THIRD_REF=dict(path=THIRD_ROOTS[0]+'/controls/third-qualification-accounting031.json',
    sha256='f1b33f36d6be915455cdb76defd085d7fe666ae3db9d702d02c63696b36ebdf1',size=31088)
THIRD_MANIFEST='30ebaf1b9416efba93072b200be7e9be5bd8ecadf8568e4f16c070f29d4c5442'
NATIVE_COUNTS=dict(constructor_worlds=2,reset_worlds=11,native_autoreset_worlds=3,world_transitions=38,learned_forwards=0,updates=0)
FIRST=1793500613
GPU_DEADLINE=1793529413
LATEST_QUALIFIER_END=FIRST+28800-16200-180-600
SOURCE_CAP=1024*1024**2
CONTINGENCY_CAP=256*1024**2
FAILED_JOBS=('49072485','49072528','49072580')
SECOND_MANIFEST='c1a57d5b566d164dd4315ecb0abf88b4ff1e881dd9cedf57ba2b9ce06f753e5e'
ZERO_COUNTS=dict.fromkeys(('constructor_worlds','learned_forwards','native_autoreset_worlds',
                           'reset_worlds','updates','world_transitions'),0)
# Exact observed private directory structure; empty cache/home directories are
# not model starts, but an unexpected empty gate directory is still rejected.
FAILED_DIRECTORIES={'claims','intents','qualification','qualification/gate-0',
    'qualification/gate-0-private',*( 'qualification/gate-0-private/'+name for name in
        ('home','tmp','tmp/torchinductor_eunikhilr','cache','cache/cuda','cache/triton',
         'cache/torch','cache/mpl','cache/warp','pycache','wandb'))}


def origin_spec():
    return dict(origin=dict(ORIGIN_REF),second_origin=dict(SECOND_REF),third_origin=dict(THIRD_REF),
        first_allocation_epoch=FIRST,original_job_id=FAILED_JOBS[0],second_failed_job_id=FAILED_JOBS[1],third_failed_job_id=FAILED_JOBS[2],
        preserved_roots=list(PRESERVED_ROOTS),aggregate_qualification_seconds=3240,
        consumed_qualification_seconds=466,remaining_qualification_seconds=2774,
        replacement_allocation_seconds=2640,latest_qualifier_end_epoch=LATEST_QUALIFIER_END,
        further_admission=False)


def _authenticate_body(record,ref,label):
    body=canonical(record)+b'\n'
    need(len(body)==ref['size'] and hashlib.sha256(body).hexdigest()==ref['sha256'],label+' evidence changed')


def validate_origin_record(record):
    _authenticate_body(record,ORIGIN_REF,'original first allocation')
    need(record['first_allocation_epoch']==FIRST and record['elapsed_gpu_seconds']==39
         and record['accounting']['job_id']==FAILED_JOBS[0] and record['accounting']['state']=='FAILED'
         and record['remote_gate_claims']==0 and record['remote_model_or_physics_starts']==0,
         'original allocation accounting differs')
    return record


def validate_second_record(record):
    _authenticate_body(record,SECOND_REF,'second technical allocation')
    need(record['schema']=='z1_depth_second_technical_allocation031'
         and record['accounting']['job_id']==FAILED_JOBS[1] and record['accounting']['state']=='FAILED'
         and record['accounting']['exit_code']=='1:0' and record['elapsed_gpu_seconds']==151
         and record['accounting']['elapsed_gpu_seconds']==151
         and record['accounting']['start_epoch']==1793503232 and record['accounting']['end_epoch']==1793503383
         and record['first_allocation_epoch']==FIRST and record['actual_start_counts']==ZERO_COUNTS
         and record['remote_model_or_physics_starts']==0 and record['remote_gate_claims']==1
         and record['remote_child_claims']==1 and len(record['failed_gate_files'])==14
         and record['manifest_sha256']==SECOND_MANIFEST
         and record['qualification_receipt_present'] is False and record['training_evaluation_outputs_empty'] is True,
         'second failed technical accounting differs')
    return record


def validate_third_record(record):
    _authenticate_body(record,THIRD_REF,'third native failed qualification')
    need(record['schema']=='z1_depth_third_qualification_accounting031'
         and record['accounting']['job_id']==FAILED_JOBS[2] and record['accounting']['state']=='FAILED'
         and record['accounting']['exit_code']=='1:0' and record['elapsed_gpu_seconds']==276
         and record['accounting']['elapsed_gpu_seconds']==276
         and record['accounting']['start_epoch']==1793505650 and record['accounting']['end_epoch']==1793505926
         and record['first_allocation_epoch']==FIRST and record['actual_start_counts']==NATIVE_COUNTS
         and record['native_environment_constructor_worlds']==2 and record['public_reset_world_requests']==11
         and record['native_in_step_reset_worlds']==3 and record['world_transitions']==38
         and record['declared_public_reset_world_requests']==10 and record['count_contract_failed'] is True
         and record['canonical_assertion_marker_present'] is True and record['child_result_present'] is False
         and record['final_compiler_import_deadline_closure_unestablished'] is True
         and record['qualification_authority'] is False and record['qualification_receipt_present'] is False
         and record['remote_gate_claims']==1 and record['remote_child_claims']==1
         and record['training_evaluation_outputs_empty'] is True and len(record['failed_files'])==118
         and record['manifest_sha256']==THIRD_MANIFEST,'third native failed qualification accounting differs')
    return record


def authenticate_third():
    return validate_third_record(read_json(verify_ref(THIRD_REF)))


def authenticate_second():
    return validate_second_record(read_json(verify_ref(SECOND_REF)))


def authenticate_origin():
    first=validate_origin_record(read_json(verify_ref(ORIGIN_REF)))
    authenticate_second();authenticate_third()  # All three exact bodies precede every new clock/claim.
    return first


def validate_clock(clock,manifest_sha=None):
    authenticate_origin()
    need(clock.get('schema')=='z1_depth_global_clock031' and clock.get('clock_domain')=='Vega_Slurm_epoch'
         and clock.get('first_allocation_epoch')==FIRST and clock.get('deadline_epoch')==GPU_DEADLINE
         and clock.get('qualification_job_id')==FAILED_JOBS[0] and clock.get('origin')==ORIGIN_REF
         and clock.get('second_origin')==SECOND_REF and clock.get('third_origin')==THIRD_REF
         and clock.get('prior_native_qualification')==dict(job_id=FAILED_JOBS[2],start_epoch=1793505650)
         and clock.get('prior_technical_allocation')==dict(job_id=FAILED_JOBS[1],start_epoch=1793503232)
         and 'replacement_allocation' not in clock,'immutable original031/three-failure clock differs')
    if manifest_sha is not None:need(clock.get('manifest_sha256')==manifest_sha,'clock release differs')
    allocation=clock.get('technical_allocation',{})
    need(set(allocation)=={'job_id','start_epoch'} and valid_replacement_start(**allocation),
         'fourth technical allocation identity differs')
    return clock


def valid_replacement_start(job_id,start_epoch):
    return (type(job_id) is str and re.fullmatch('[1-9][0-9]*',job_id) and job_id not in FAILED_JOBS
            and type(start_epoch) in (int,float) and math.isfinite(start_epoch)
            and 1793505926<start_epoch<=LATEST_QUALIFIER_END-2640)


def admit_replacement(allocation,*,now):
    authenticate_origin()
    start,end=allocation['start_epoch'],allocation['end_epoch']
    need(valid_replacement_start(allocation['job_id'],start),'fourth technical start/origin bound differs')
    need(all(type(x) in (int,float) and math.isfinite(x) for x in (end,now)), 'finite technical bounds required')
    need(end-start==2640 and 466+(end-start)<=3240,'aggregate qualification budget exceeded')
    need(start<=now and now+2470<=end and end<=LATEST_QUALIFIER_END,
         'HOLD_UNCLAIMED: qualification/slot/review time cannot fit')
    return end


def _empty(path,*,directories=False):
    """Only private/outputs may contain harmless empty directory skeletons."""
    from .lifecycle import tree_bytes
    safe_path(path)
    if not path.exists():return
    need(path.is_dir() and not path.is_symlink(),'unused-label directory differs')
    tree_bytes(path)
    need(not any(p.is_file() or not directories for p in path.rglob('*')),
         'seed labels are not unused: prior scientific claims/checkpoints/files')


def _failed_technical_tree(root,record):
    from .lifecycle import tree_bytes
    actual={};directories=set()
    for name in ('claims','intents','qualification'):
        path=root/name
        need(path.is_dir() and not path.is_symlink(),'failed technical directory missing')
        tree_bytes(path)  # Reject symlink/special entries before regular inventory.
        directories.add(name)
        directories.update(str(p.relative_to(root)) for p in path.rglob('*') if p.is_dir())
        actual.update({name+'/'+rel:ref for rel,ref in inventory(path,1024**3).items()})
    need(actual==record['failed_gate_files'] and directories==FAILED_DIRECTORIES,
         'retained failed technical inventory/structure differs')
    ref=record['carried_clock'];verify_ref(dict(ref,path=str(root/'clock.json')))
    clock=read_json(root/'clock.json')
    need(clock['first_allocation_epoch']==FIRST and clock['deadline_epoch']==GPU_DEADLINE
         and clock['qualification_job_id']==FAILED_JOBS[0]
         and clock['replacement_allocation']==dict(job_id=FAILED_JOBS[1],start_epoch=1793503232),
         'retained failed technical clock differs')
    need(digest(read_json(root/'manifest.json'))==SECOND_MANIFEST,'retained failed technical manifest differs')
    terminal=read_json(root/'qualification/gate-0/terminal.json')
    need(terminal['status']=='FAILED_NO_RETRY' and terminal['kind']=='qualification' and terminal['index']==0
         and terminal['manifest_sha256']==SECOND_MANIFEST
         and read_json(root/'qualification/gate-0/actual-start-counts.json')==ZERO_COUNTS,
         'retained failed terminal/counts differ')


def _failed_native_tree(root,record):
    """Exact native failure bytes; marker alone never establishes qualification."""
    from .lifecycle import tree_bytes
    expected=record['failed_files'];actual={};directories=set()
    allowed=set(FAILED_DIRECTORIES)|{'slurm'}
    for rel in expected:
        allowed.update(str(p) for p in Path(rel).parents if str(p)!='.')
    for name in ('claims','intents','qualification','slurm'):
        path=root/name
        need(path.is_dir() and not path.is_symlink(),'native failed directory missing')
        tree_bytes(path)
        directories.add(name)
        directories.update(str(p.relative_to(root)) for p in path.rglob('*') if p.is_dir())
        actual.update({name+'/'+rel:ref for rel,ref in inventory(path,1024**3).items()})
    need(directories<=allowed,'native failed directory structure differs')
    for name in ('clock.json','manifest.json','qualification-grant.json','qualification-recipe.json'):
        path=verify_ref(dict(expected[name],path=str(root/name)))
        actual[name]={key:expected[name][key] for key in ('sha256','size')}
    need(actual==expected,'retained native failed inventory differs')
    clock=read_json(root/'clock.json')
    need(clock['first_allocation_epoch']==FIRST and clock['deadline_epoch']==GPU_DEADLINE
         and clock['qualification_job_id']==FAILED_JOBS[0]
         and clock['origin']==dict(ORIGIN_REF,path=FIRST_ROOTS[0]+'/controls/first-gpu-allocation-accounting031.json')
         and clock['second_origin']==dict(SECOND_REF,path=SECOND_ROOTS[0]+'/controls/second-technical-allocation-accounting031.json')
         and clock['prior_technical_allocation']==dict(job_id=FAILED_JOBS[1],start_epoch=1793503232)
         and clock['technical_allocation']==dict(job_id=FAILED_JOBS[2],start_epoch=1793505650),
         'retained native failed clock differs')
    need(digest(read_json(root/'manifest.json'))==THIRD_MANIFEST,'retained native failed manifest differs')
    terminal=read_json(root/'qualification/gate-0/terminal.json')
    need(terminal['status']=='FAILED_NO_RETRY' and terminal['kind']=='qualification' and terminal['index']==0
         and terminal['manifest_sha256']==THIRD_MANIFEST
         and read_json(root/'qualification/gate-0/actual-start-counts.json')==NATIVE_COUNTS,
         'native failed terminal/counts differ')
    need(not (root/'qualification/gate-0/child-result.json').exists(),'native failed child unexpectedly closed')


def prove_unused():
    """Only policy/evaluation labels are unused; gate0/child are consumed."""
    authenticate_origin();record=authenticate_second();native=authenticate_third()
    for index,name in enumerate(PRESERVED_ROOTS):
        root=safe_path(name);need(root.is_dir(),'preserved namespace missing')
        need(not safe_path(root/'qualification-receipt.json').exists(),'failed technical qualification receipt present')
        # Closed observed top-level layout for these six retained roots only.
        # Subsets support relocated fixtures; the proofs below check completeness.
        directories={'claims','intents','outputs','private'}
        files=set()
        if index%2==0:
            directories.update(('controls','qualification','safe_impact_manipulation','slurm','source','submissions'))
            files.update(('asset-copy-plan.json','manifest.json','qualification-grant.json','qualification-recipe.json'))
            if index in (2,4):files.add('clock.json')
        for path in root.iterdir():
            safe_path(path)
            mode=path.lstat().st_mode
            need((path.name in directories and stat.S_ISDIR(mode))
                 or (path.name in files and stat.S_ISREG(mode)),
                 'unexpected preserved root entry name/type: '+str(path))
        for name in ('outputs','private'):_empty(root/name,directories=index>=2)
        if index==2:_failed_technical_tree(root,record)
        elif index==4:_failed_native_tree(root,native)
        else:
            for name in ('claims','intents','qualification'):_empty(root/name)
    return dict(training_evaluation_claims_checkpoints=0,gate_claims=2,child_claims=2,
        prior_native_work={k:NATIVE_COUNTS[k] for k in ('constructor_worlds','reset_worlds','native_autoreset_worlds','world_transitions')},
        failed_technical_job_ids=list(FAILED_JOBS),scope=list(PRESERVED_ROOTS),qualification_authority=False,
        labels='technical gate0/child consumed; policy/evaluation labels never trained')


def scope_usage(*,additional_inputs=0,contingency_liabilities=0):
    """Count eight roots once; reserve new qualification and charge old failures."""
    from .lifecycle import tree_bytes
    need(type(additional_inputs) is int and additional_inputs>=0,'additional input bytes invalid')
    need(type(contingency_liabilities) is int and contingency_liabilities>=0,'contingency liabilities invalid')
    total=inputs=failed=0
    for index,name in enumerate(SCOPE_ROOTS):
        root=safe_path(name)
        if index<6:need(root.is_dir() and not root.is_symlink(),'preserved namespace missing')
        if not root.exists():continue
        total+=tree_bytes(root)
        for path in root.iterdir():
            if path.name in ('outputs','private','qualification'):
                if index<6 and path.name=='qualification':failed+=tree_bytes(path)
                continue
            inputs+=tree_bytes(path) if path.is_dir() else path.stat().st_size
    need(inputs+additional_inputs<=SOURCE_CAP,'aggregate preserved/successor source/assets/controls cap exceeded')
    # The full new qualification reservation is retained; old evidence is not
    # credited against it or subtracted a second time from the overall cap.
    need(failed+contingency_liabilities<=CONTINGENCY_CAP,'failed technical evidence/contingency liabilities exceed cap')
    return dict(total_bytes=total,source_assets_controls_bytes=inputs,failed_qualification_bytes=failed,
        contingency_liabilities_bytes=contingency_liabilities,
        remaining_contingency_bytes=CONTINGENCY_CAP-failed-contingency_liabilities)
