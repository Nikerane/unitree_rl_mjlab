"""Closed terminal archive path contract; no filesystem/network execution on import."""
from pathlib import Path
import re

CAMPAIGN='/cephhome/eunikhilr/campaigns/z1-horizon-training-20261023-v1'
SHARED_SHA='3d16a8dbc078f6b904c0b713aa5c8188753cf624c1914078a462f2df358e017d'
SEEDS=(2026102301,2026102302,2026102303)
ARMS=('native50','extended200')
BANKS=(20261041,2026101601)
BASE='Unitree-Z1-Hammer-Quintic-ThreeStarts-SplineSpeedCurriculum-Observation-Age-LowNoise-DeferredCredit'

def require(condition,message):
    if not condition:raise ValueError(message)

def policy_identity(policy):
    require(isinstance(policy,dict),'policy object required')
    arm=policy.get('arm');seed=policy.get('training_seed')
    require(arm in ARMS and type(seed)is int and seed in SEEDS,'outside exact six policies')
    require(policy.get('policy_id')==f'{arm}-seed{seed}' and policy.get('task')==BASE+'-Horizon-'+arm
        and policy.get('native_window_s')=={ 'native50':.05,'extended200':.2}[arm],'policy identity mismatch')
    return policy['policy_id']

def plain_absolute(value):
    path=Path(value)
    require(path.is_absolute() and '..' not in path.parts and str(path)==value,'canonical absolute path required')
    return path

def validate_layout(layout):
    require(isinstance(layout,dict) and layout.get('schema')=='z1_horizon_cpu_archive_layout_v1','closed CPU layout required')
    root=plain_absolute(layout['root'])
    require(root==Path('/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002'),'exact predeclared CPU evaluation root required')
    rows=layout.get('slots',[]);expected=set();seen=set()
    for seed in SEEDS:
        for arm in ARMS:
            for scope,bank in [('own_cpu32',None),('science160',BANKS[0]),('science160',BANKS[1])]:expected.add((f'{arm}-seed{seed}',scope,bank))
    require(len(rows)==18,'exact eighteen CPU paths required')
    for row in rows:
        require(set(row)=={'policy_id','scope','bank','run_root','private_root'},'exact CPU layout row required')
        key=(row['policy_id'],row['scope'],row['bank']);require(key in expected and key not in seen,'foreign/duplicate CPU slot')
        seen.add(key);tag='own_cpu32' if row['scope']=='own_cpu32' else 'science160-bank'+str(row['bank'])
        if row['scope']=='own_cpu32':tag+='-staged-integration002'
        require(row['run_root']==str(root/'runs'/row['policy_id']/tag) and row['private_root']==str(root/'private'/row['policy_id']/tag),'CPU slot path not canonical')
    require(seen==expected,'CPU layout incomplete')
    return layout

def validate_paths(request):
    name=policy_identity(request['policy'])
    require(request['policy_id']==name and request['campaign_root']==CAMPAIGN,'foreign training campaign/policy')
    scope=request['scope'];bank=request.get('bank')
    if scope=='training':
        require(bank is None and request.get('cpu_layout') is None,'training must not select CPU layout/bank')
        run=Path(CAMPAIGN)/'runs'/name;private=Path(CAMPAIGN)/'private'/name
    else:
        require(scope in ('own_cpu32','science160') and (bank is None if scope=='own_cpu32' else type(bank)is int and bank in BANKS),'scope/bank outside closed allocation')
        layout=validate_layout(request['cpu_layout'])
        row=next(row for row in layout['slots'] if (row['policy_id'],row['scope'],row['bank'])==(name,scope,bank))
        run=Path(row['run_root']);private=Path(row['private_root'])
    require(request['source_root']==str(run) and request['private_root']==str(private),'requested roots differ from canonical mapping')
    require(re.fullmatch(r'[0-9]+(?:_[0-9]+)?',request['job_id']) is not None,'exact terminal Slurm job required')
    return {'policy-tree':run,'private-tree':private}


def accounting_identity(job_id,accounting_job_id,raw_id,control):
    """Prove genuine Slurm array alias to concrete allocation; never rewrite rows."""
    require(re.fullmatch(r'[0-9]+',job_id)is not None and raw_id==job_id,'exact concrete JobIDRaw required')
    require(isinstance(control,dict) and control.get('argv') in (['scontrol','show','job',job_id,'-o'],['scontrol','show','job',accounting_job_id,'-o'])
        and type(control.get('returncode'))is int and control['returncode']==0,'exact successful scontrol query required')
    for key in ('stdout','stderr'):
        require(isinstance(control.get(key),str) and len(control[key].encode())<=1024**2,'bounded genuine control response required')
    lines=[line for line in control['stdout'].splitlines()if line.strip()]
    require(len(lines)==1,'single actual control allocation required')
    pairs=re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)',lines[0])
    require(len({key for key,_ in pairs})==len(pairs),'duplicate control keys rejected')
    fields=dict(pairs);require(fields.get('JobId')==job_id,'scontrol concrete JobId differs')
    alias=re.fullmatch(r'([0-9]+)_([0-9]+)',accounting_job_id)
    if alias:
        require(fields.get('ArrayJobId')==alias.group(1) and fields.get('ArrayTaskId')==alias.group(2),'actual array identity differs or missing')
    else:
        require(accounting_job_id==job_id and not fields.get('ArrayJobId') and not fields.get('ArrayTaskId'),'nonarray exact identity required')
    require(isinstance(fields.get('JobState'),str) and fields['JobState'],'actual captured control state required')
    return dict(job_id=job_id,accounting_job_id=accounting_job_id,job_id_raw=raw_id,
        array_job_id=fields.get('ArrayJobId'),array_task_id=fields.get('ArrayTaskId'),control_state_at_capture=fields['JobState'],identity_only=True)


def validate_identity_capture(value,source_artifact):
    require(isinstance(value,dict) and value.get('schema')=='z1_slurm_identity_capture_v1','explicit actual identity capture required')
    require(value.get('capture')==source_artifact.get('scheduler',{}).get('scontrol')
        and value.get('observed_epoch')==source_artifact.get('observed_epoch'),'identity differs from hash-bound genuine source artifact')
    require(type(value.get('observed_epoch')) in (int,float),'actual capture epoch required')
    return accounting_identity(value['job_id'],value['accounting_job_id'],value['job_id'],value['capture'])
