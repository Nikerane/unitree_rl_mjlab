"""Closed thirty-policy proposal and authority; standard library only."""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

ROOT='/cephhome/eunikhilr/campaigns/z1-training-thirty-20261030-v2'
BASE_ROOT='/cephhome/eunikhilr/campaigns/z1-exploration-pilot-20261029-seed2026102902-v2'
BASE_SHA='d87045ebd5f776ec27d59d7d3958b3d1a58b51ea3cb9d4d47ff4c8a16e2d53dd'
PYTHON='/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python'
RESOLVED_PYTHON='/cephhome/eunikhilr/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'
PYTHON_SHA='d778d0b551287fb5f06c4464f78b616c086f8d42e94d8cfc209b12324cf937a1'
MIB=1024**2
LIMITS=dict(total_bytes=7936*MIB,quota_floor=1024**3,output_bytes=128*MIB,
    private_bytes=96*MIB,control_bytes=384*MIB,qualification_bytes=512*MIB,
    reserve_bytes=256*MIB,trace_bytes=96*MIB,population_bytes=16*MIB,
    case_seconds=7200,allocation_seconds=32400,cleanup_seconds=60,
    qualification_seconds=2700,gate_seconds=720,gate_private_bytes=128*MIB,gate_log_bytes=16*MIB)
FAMILIES={'timing_noise':'scripts.z1_timing_noise_training020',
          'critic_noise':'scripts.z1_critic_noise_training021',
          'split_credit':'scripts.z1_split_credit_training021'}

def need(ok,message):
    if not ok:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def regular_path(path):
    p=Path(path);need(p.is_absolute(),'absolute path required')
    for q in (p,*p.parents):need(not q.is_symlink(),'symlink forbidden')
    need(stat.S_ISREG(p.stat().st_mode),'regular file required');return p
def check_ref(ref):
    need(type(ref) is dict and set(ref)=={'path','sha256','size'},'reference fields differ')
    p=Path(ref['path']);need(p.is_absolute() and str(p)==ref['path'] and '..' not in p.parts,'reference path differs')
    need(type(ref['sha256']) is str and re.fullmatch('[0-9a-f]{64}',ref['sha256']),'invalid SHA256')
    need(type(ref['size']) is int and 0<ref['size']<=LIMITS['total_bytes'],'invalid reference size')
def expected_cells():
    cells=[]
    for index in range(30):
        family='timing_noise' if index<12 else 'critic_noise' if index<24 else 'split_credit'
        local=index if index<12 else index-12 if index<24 else index-24
        width=2 if family=='split_credit' else 4
        slot=(0 if index<12 else 3 if index<24 else 6)+local//width
        cell=dict(index=local,setting=('vic_baseline','vic_position_low')[local%2],seed=2026103101+local//width,
                  critic_mode='event' if family=='critic_noise' and local%4>=2 else 'current',updates=500,worlds=1024,rollout_steps=24)
        if family=='split_credit':cell.update(early_fraction=.5,early_deadline_ms=50,reserve_deadline_ms=200)
        else:cell['credit_deadline_ms']=100 if family=='critic_noise' else 50 if local%4<2 else 200
        name=f'cell-{index:02d}'
        cells.append(dict(index=index,family=family,training_index=local,slot=slot,training_cell=cell,
            output=f'{ROOT}/outputs/{name}',private=f'{ROOT}/private/{name}',claim=f'{ROOT}/claims/{name}.json'))
    return cells
def reference_index(cell):
    if cell['family']=='split_credit':return (cell['slot']-6)*4+cell['training_index']%2
    return cell['index']-cell['training_index']%4

def slot_cells(release,slot):
    need(type(slot) is int and 0<=slot<9,'slot outside finite nine')
    return [c for c in release['cells'] if c['slot']==slot]
def required_free(used):
    need(type(used) is int and 0<=used<=LIMITS['total_bytes'],'aggregate byte limit exceeded')
    return LIMITS['quota_floor']+LIMITS['total_bytes']-used

def build_proposal(metadata,source_files,asset_files,qualification_recipe_ref,split_equivalence_ref=None):
    need(type(metadata) is dict and metadata.get('base_root')==BASE_ROOT,'frozen base root differs')
    base=metadata['manifest'];need(digest(base)==BASE_SHA,'frozen base manifest differs')
    need(len(base['source_files'])==660 and len(base['runtime_files'])==375 and len(base['asset_files'])==32,'base inventory counts differ')
    need(type(source_files) is dict and all(p in source_files and (p=='src/tasks/hammer/rl/impact_speed_contract.py' or source_files[p]==h) for p,h in base['source_files'].items()),'base scientific source changed')
    for path,sha in source_files.items():
        p=Path(path);need(not p.is_absolute() and '..' not in p.parts and str(p)==path,'invalid source member')
        need(type(sha) is str and re.fullmatch('[0-9a-f]{64}',sha),'invalid source SHA')
    expected_assets={str(Path(ROOT)/Path(p).relative_to(BASE_ROOT)):sha for p,sha in base['asset_files'].items()}
    need(asset_files==expected_assets,'unchanged sibling asset layout required')
    required=['scripts/z1_timing_noise020/'+n for n in ('__init__.py','admission.py','processes.py','worker.py','artifacts.py','compiler.py','__main__.py','long_credit_trace.py','training.sbatch','qualification.sbatch','split_contract.py')]
    required += [name.replace('.','/')+'.py' for name in FAMILIES.values()]
    need(all(p in source_files for p in required),'full thirty-policy overlay inventory required')
    check_ref(qualification_recipe_ref)
    need(Path(qualification_recipe_ref['path']).is_relative_to(ROOT),'qualification recipe outside control root')
    if split_equivalence_ref is not None:
        check_ref(split_equivalence_ref)
        need(Path(split_equivalence_ref['path']).is_relative_to(ROOT),'split proof outside counted control root')
    from .compiler import policy
    return json.loads(canonical(dict(schema='z1_thirty_training_release020',owner='root',scope='training30_500',
        root=ROOT,source_root=ROOT+'/source',metadata=metadata,source_files=source_files,asset_files=asset_files,
        runtime_files=base['runtime_files'],runtime_roles=base['runtime_roles'],compiler_policy=policy(),
        base_overlay={p:source_files[p] for p in ('src/tasks/hammer/rl/impact_speed_contract.py',) if source_files[p]!=base['source_files'][p]},
        qualification_recipe_ref=qualification_recipe_ref,split_equivalence_ref=split_equivalence_ref,
        cells=expected_cells(),limits=LIMITS,python=dict(path=PYTHON,resolved=RESOLVED_PYTHON,sha256=PYTHON_SHA),
        allocation=dict(cpus=4,memory_mib=32768,gpus=1,threads=4,max_active_slots=6),retry=False,requeue=False,resume=False)))

@dataclass(frozen=True)
class ValidatedRelease:
    canonical_json:bytes
    authorized:bool
    def data(self):return json.loads(self.canonical_json)
def validate_approval(release,approval,now):
    need(type(now) in (int,float) and math.isfinite(now) and now>=0,'invalid clock')
    need(type(approval) is dict and set(approval)=={'schema','owner','scope','manifest_sha256','issued_epoch','expires_epoch'},'approval fields differ')
    need(approval['schema']=='z1_thirty_training_approval020' and approval['owner']=='root' and approval['scope']=='training30_500','root training authority required')
    need(approval['manifest_sha256']==digest(release),'authority manifest mismatch')
    issued,expiry=approval['issued_epoch'],approval['expires_epoch']
    need(all(type(v) in (int,float) and math.isfinite(v) for v in (issued,expiry)),'authority clock invalid')
    need(0<=issued<=now<expiry and 0<expiry-issued<=72*3600,'expired/future/overlong authority')
def validate_release(release,now,approval=None):
    expected=build_proposal(*(release[k] for k in ('metadata','source_files','asset_files','qualification_recipe_ref','split_equivalence_ref')))
    need(canonical(release)==canonical(expected),'closed release differs')
    if approval is not None:
        validate_approval(release,approval,now)
        need(release['split_equivalence_ref'] is not None,'split equivalence proof missing')
    return ValidatedRelease(canonical(release),approval is not None)
def inventory_for(release):
    refs={}
    for kind in ('source','runtime','asset'):
        for path,sha in release[kind+'_files'].items():
            absolute=str(Path(release['source_root'])/path) if kind=='source' else path
            refs[kind+':'+path]=dict(path=absolute,sha256=sha)
    for path,sha in release['compiler_policy']['runtime_files'].items():
        refs['compiler:'+path]=dict(path=path,sha256=sha)
    refs['python']=release['python'];refs['recipe']=release['qualification_recipe_ref']
    if release['split_equivalence_ref']:refs['split_equivalence']=release['split_equivalence_ref']
    return refs
def verify_inventory(refs):
    need(type(refs) is dict and refs,'inventory missing')
    for name,ref in refs.items():
        if name=='python':
            need(ref==dict(path=PYTHON,resolved=RESOLVED_PYTHON,sha256=PYTHON_SHA),'Python inventory differs')
            need(str(Path(PYTHON).resolve(strict=True))==RESOLVED_PYTHON,'Python alias differs');p=regular_path(RESOLVED_PYTHON)
        else:p=regular_path(ref['path'])
        if 'size' in ref:need(p.stat().st_size==ref['size'],'inventory size differs')
        need(sha_file(p)==ref['sha256'],'inventory SHA differs: '+str(p))
    return True
def verify_source_closure(release):
    root=Path(release['source_root']);actual=set()
    def failed(exc):raise exc
    need(root.is_dir() and not root.is_symlink(),'source root required')
    for directory,dirs,files in os.walk(root,followlinks=False,onerror=failed):
        for name in dirs:need(not (Path(directory)/name).is_symlink(),'source symlink directory')
        for name in files:
            p=regular_path(Path(directory)/name);actual.add(str(p.relative_to(root)))
    need(actual==set(release['source_files']),'source member closure differs')
def validate_allocation(*,environ,hostname,affinity,executable,resolved_executable,hash_fn):
    need(re.fullmatch(r'gn(?:0[1-9]|[1-5][0-9]|60)',hostname),'GPU node outside gn01..gn60')
    need(environ.get('SLURM_CPUS_PER_TASK')=='4' and environ.get('SLURM_MEM_PER_NODE')=='32768' and len(affinity)==4,'allocation CPU/memory/affinity differs')
    need(environ.get('SLURM_GPUS_ON_NODE')=='1','exactly one allocated GPU required')
    need(all(environ.get(k)=='4' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'thread limits differ')
    need(executable==PYTHON and resolved_executable==RESOLVED_PYTHON and hash_fn(resolved_executable)==PYTHON_SHA,'Python invocation/hash differs')
    need(re.fullmatch('[1-9][0-9]*',environ.get('SLURM_JOB_ID','')),'Slurm job required')
def claim_once(path,record):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(canonical(record)+b'\n');stream.flush();os.fsync(stream.fileno())
def read_json(path,cap=4*1024**2):
    p=regular_path(path);need(p.stat().st_size<=cap,'JSON byte cap exceeded')
    def pairs(items):
        out={}
        for key,value in items:need(key not in out,'duplicate JSON key');out[key]=value
        return out
    return json.loads(p.read_bytes(),object_pairs_hook=pairs,parse_constant=lambda _:need(False,'nonfinite JSON'))
