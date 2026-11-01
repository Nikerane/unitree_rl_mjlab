"""The single reviewed031 successor: immutable origin and retained scope only."""
import math
import re
from pathlib import Path
from .files import canonical, digest, need, read_json, verify_ref

PRESERVED_ROOTS=('/cephhome/eunikhilr/campaigns/z1-depth-reward-20261101-v1',
                 '/cephhome/eunikhilr/campaigns/z1-depth-evaluation-20261101-v1')
SUCCESSOR_ROOTS=tuple(root[:-1]+'2' for root in PRESERVED_ROOTS)
SCOPE_ROOTS=PRESERVED_ROOTS+SUCCESSOR_ROOTS
ORIGIN_REF=dict(path=PRESERVED_ROOTS[0]+'/controls/first-gpu-allocation-accounting031.json',
    sha256='b45b58d58a7f8634201bc33b6b12dcd32900f867251a144d64755be12e543a20',size=1557)
FIRST=1793500613
GPU_DEADLINE=1793529413
LATEST_QUALIFIER_END=FIRST+28800-16200-180-600
SOURCE_CAP=512*1024**2


def origin_spec():
    return dict(origin=dict(ORIGIN_REF),first_allocation_epoch=FIRST,original_job_id='49072485',
        preserved_roots=list(PRESERVED_ROOTS),aggregate_qualification_seconds=2700,
        consumed_qualification_seconds=39,remaining_qualification_seconds=2661,
        replacement_allocation_seconds=2640,latest_qualifier_end_epoch=LATEST_QUALIFIER_END)


def validate_origin_record(record):
    import hashlib
    body=canonical(record)+b'\n'
    need(len(body)==ORIGIN_REF['size'] and hashlib.sha256(body).hexdigest()==ORIGIN_REF['sha256'],
         'original first allocation evidence changed')
    need(record['first_allocation_epoch']==FIRST and record['elapsed_gpu_seconds']==39
         and record['accounting']['job_id']=='49072485' and record['remote_gate_claims']==0
         and record['remote_model_or_physics_starts']==0,'original allocation accounting differs')
    return record


def authenticate_origin():
    return validate_origin_record(read_json(verify_ref(ORIGIN_REF)))


def validate_clock(clock,manifest_sha=None):
    authenticate_origin()
    need(clock.get('schema')=='z1_depth_global_clock031' and clock.get('clock_domain')=='Vega_Slurm_epoch'
         and clock.get('first_allocation_epoch')==FIRST and clock.get('deadline_epoch')==GPU_DEADLINE
         and clock.get('qualification_job_id')=='49072485' and clock.get('origin')==ORIGIN_REF,
         'immutable original031 clock differs')
    if manifest_sha is not None:need(clock.get('manifest_sha256')==manifest_sha,'clock release differs')
    replacement=clock.get('replacement_allocation',{})
    need(set(replacement)=={'job_id','start_epoch'} and valid_replacement_start(**replacement),
         'replacement allocation identity differs')
    return clock


def valid_replacement_start(job_id,start_epoch):
    return (type(job_id) is str and re.fullmatch('[1-9][0-9]*',job_id) and job_id!='49072485'
            and type(start_epoch) in (int,float) and math.isfinite(start_epoch)
            and FIRST<=start_epoch<=LATEST_QUALIFIER_END-2640)


def admit_replacement(allocation,*,now):
    authenticate_origin()
    start,end=allocation['start_epoch'],allocation['end_epoch']
    need(valid_replacement_start(allocation['job_id'],start),'replacement start/origin bound differs')
    need(all(type(x) in (int,float) and math.isfinite(x) for x in (end,now)), 'finite replacement bounds required')
    need(end-start==2640 and 39+(end-start)<=2700,'aggregate qualification budget exceeded')
    need(start<=now and now+2470<=end and end<=LATEST_QUALIFIER_END,
         'HOLD_UNCLAIMED: qualification/slot/review time cannot fit')
    return end


def prove_unused():
    """Labels exist in technical metadata; no prior consumed science is allowed."""
    from .lifecycle import tree_bytes
    for root in PRESERVED_ROOTS:
        root=Path(root);need(root.is_dir() and not root.is_symlink(),'preserved namespace missing')
        for name in ('claims','intents','outputs','private','qualification'):
            path=root/name
            if path.exists():
                tree_bytes(path)  # reject symlinks/special files, including zero bytes
                need(not any(path.iterdir()),'seed labels are not unused: prior claims/intents/scientific files')
    return dict(training_evaluation_claims_checkpoints=0,gate_claims=0,scope=list(PRESERVED_ROOTS),
                labels='present in technical release metadata; never trained')


def scope_usage(*,additional_inputs=0):
    """Count each retained/successor byte once, including both input trees."""
    from .lifecycle import tree_bytes
    need(type(additional_inputs) is int and additional_inputs>=0,'additional input bytes invalid')
    total=inputs=0
    for index,name in enumerate(SCOPE_ROOTS):
        root=Path(name)
        if index<2:need(root.is_dir() and not root.is_symlink(),'preserved namespace missing')
        if not root.exists():continue
        total+=tree_bytes(root)
        for path in root.iterdir():
            # Scientific directories have their own existing component bounds.
            if path.name in ('outputs','private','qualification'):continue
            inputs+=tree_bytes(path) if path.is_dir() else path.stat().st_size
    need(inputs+additional_inputs<=SOURCE_CAP,'aggregate preserved/successor source/assets/controls cap exceeded')
    return dict(total_bytes=total,source_assets_controls_bytes=inputs)
