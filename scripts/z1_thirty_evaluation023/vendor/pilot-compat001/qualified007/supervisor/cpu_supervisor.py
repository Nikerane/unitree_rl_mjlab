"""Candidate finite CPU evaluation supervisor. No actual release or jobs supplied."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import types

SEEDS=(2026102301,2026102302,2026102303)
ARMS=('native50','extended200')
BANKS=(20261041,2026101601)
CASES=['original-route-1','start-minus10mm','original-route+0','start-plus10mm','original-route+1']
CAMPAIGN_ROOT='/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002'
CLAIM_ROOT='/cephhome/eunikhilr/campaigns/.horizon-cpu-once'

CURRENT_PACKAGE=Path(CAMPAIGN_ROOT)/'package-runtime-budget007'
LEGACY_SUPERVISOR={'path':str(Path(CAMPAIGN_ROOT)/'package-staged-integration002/supervisor/cpu_supervisor.py'),
    'sha256':'565bb43027d8c9a98505603985d7a95b8f54c093c51e6cbcb2ba947f1507c4fd'}
LEGACY_WORKER={'path':str(Path(CAMPAIGN_ROOT)/'package-staged-integration002/horizon_worker.py'),
    'sha256':'9c44eb941f08084d67695d23f73977b4399e375d50a2271addd6f2873cc3efc0'}
LEGACY_FIRST_INTENT={'path':str(Path(CAMPAIGN_ROOT)/'runs/native50-seed2026102301/own_cpu32-staged-integration002/intent.json'),
    'sha256':'8e2097dea3b4f1f29db562182972c49842a476379b056f3f1084c280d0625794'}
LEGACY_FIRST_SEAL={'path':str(Path(CAMPAIGN_ROOT)/'admission-staged-integration002/first/qualification-review001/own32-qualification-seal.json'),
    'sha256':'7a057792712b75a4a288ef3565e0ba405d23ecbf71c0d037a40a3ee8d71fb16a'}
LEGACY_FIRST_BACKUP={'path':'/cephhome/eunikhilr/campaign-backups/z1-horizon-evaluation-20261027-resident004/native50-seed2026102301/own_cpu32-staged-integration002/job-48670474/consumer-receipt.json',
    'sha256':'f226830f8f46f263637ae9dee62c05872869b7b7b2dfcea4b1ae551da9948fd0'}
HISTORICAL_SLOT0_SHA256='a7c9776d15feacf18a5bdce7645f1d13f5c2ff078f4b95b340e856dcd1a8a106'
DIAGNOSTIC_INTEGRATION_SHA256='ed9d28c7781ed5bb776049d9cca52536f51c405215bd0c3d18395defc35ee3d7'

BASE='Unitree-Z1-Hammer-Quintic-ThreeStarts-SplineSpeedCurriculum-Observation-Age-LowNoise-DeferredCredit'

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def durable(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def bound(reference):
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('hashed reference required')
    path=Path(reference['path'])
    if not path.is_absolute() or path.is_symlink() or sha(path)!=reference['sha256']:raise ValueError('bound bytes changed')
    return json.loads(path.read_text())

def inventory(root):
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise ValueError('actual complete directory required')
    members={}
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('tree symlink forbidden')
        if path.is_file():members[str(path.relative_to(root))]=sha(path)
    if not members:raise ValueError('empty tree')
    return members

def tree_digest(members):return hashlib.sha256(json.dumps(members,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def finite_tree(value):
    if hasattr(value,'detach'):
        if not bool(value.detach().isfinite().all().item()):raise ValueError('nonfinite checkpoint tensor')
    elif isinstance(value,float):
        if not math.isfinite(value):raise ValueError('nonfinite checkpoint scalar')
    elif isinstance(value,dict):
        for child in value.values():finite_tree(child)
    elif isinstance(value,(list,tuple)):
        for child in value:finite_tree(child)

def audit_training(binding,loader=None,*,tensor_audit=True,runtime_intent=None):
    required={'policy','run_root','manifest','terminal','completion','config','live499','checkpoint','backup'}
    if not isinstance(binding,dict) or set(binding)!=required:raise ValueError('complete training audit bindings required')
    policy=binding['policy'];cell=dict(arm=policy['arm'],seed=policy['training_seed'],name=policy['policy_id'])
    manifest=bound(binding['manifest']);terminal=bound(binding['terminal']);completion=bound(binding['completion'])
    config=bound(binding['config']);live=bound(binding['live499']);backup=bound(binding['backup'])
    if (manifest.get('schema')!='z1_horizon_campaign_v1' or cell not in manifest.get('cells',[])
        or manifest.get('updates')!=500 or manifest.get('worlds')!=1024 or manifest.get('rollout_steps')!=24
        or manifest.get('gamma')!=.99 or manifest.get('velocity_pmax')!=.5 or manifest.get('impulse_pmax')!=.2):
        raise ValueError('training manifest recipe/cell mismatch')
    if (terminal.get('status')!='COMPLETE' or terminal.get('returncode')!=0
        or completion.get('status')!='COMPLETE' or completion.get('completed_updates')!=500
        or any(record.get('cell')!=cell for record in (completion,config))
        or config.get('task')!=policy['task'] or config.get('resume') is not False
        or config.get('source_manifest')!=binding['manifest']['sha256']
        or config.get('velocity_pmax')!=.5 or config.get('impulse_pmax')!=.2
        or config.get('horizon_s')!=policy['native_window_s']):raise ValueError('actual training terminal/config mismatch')
    if (live.get('iteration')!=499 or live.get('common_step_counter')!=12000 or live.get('last_transition_index')!=11999
        or live.get('velocity_pmax')!=.5 or live.get('impulse_pmax')!=.2 or live.get('scheduler') is not False):
        raise ValueError('actual training clock/constant pressure mismatch')
    checkpoint=Path(binding['checkpoint']['path'])
    if (not checkpoint.is_absolute() or checkpoint.is_symlink() or sha(checkpoint)!=binding['checkpoint']['sha256']
        or policy['checkpoint_path']!=str(checkpoint) or policy['checkpoint_sha256']!=sha(checkpoint)
        or completion.get('checkpoint_sha256',{}).get('model_499.pt')!=sha(checkpoint)
        or live.get('checkpoint_sha256')!=sha(checkpoint)):raise ValueError('final checkpoint identity mismatch')
    members=inventory(binding['run_root']);digest=tree_digest(members)
    for name in ('terminal','completion','config','live499','checkpoint'):
        path=Path(binding[name]['path']).resolve()
        if not path.is_relative_to(Path(binding['run_root']).resolve()):raise ValueError('training evidence escaped own tree')
    if (backup.get('status')!='PASS' or backup.get('independent_readback')!='PASS'
        or backup.get('policy')!=policy or backup.get('members')!=members or backup.get('tree_sha256')!=digest
        or backup.get('training_manifest_sha256')!=binding['manifest']['sha256']):
        raise ValueError('complete independently preserved training tree required')
    if tensor_audit:
        if loader is None:
            if runtime_intent is None:raise ValueError('actual tensor audit requires pinned runtime intent')
            import torch
            verify_runtime_module(runtime_intent,'torch',torch)
            loader=lambda path:torch.load(path,map_location='cpu',weights_only=False)
        payload=loader(checkpoint)
        if (payload.get('iter')!=499 or payload.get('infos',{}).get('env_state',{}).get('common_step_counter')!=12000
            or payload.get('infos',{}).get('cat_pressure_treatment') is not None
            or not payload.get('actor_state_dict') or not payload.get('critic_state_dict')):
            raise ValueError('actual finite checkpoint state/clock required')
        finite_tree(payload)
    return dict(status='PASS'if tensor_audit else'METADATA_PREFLIGHT_INCOMPLETE',policy=policy,tree_sha256=digest,checkpoint_sha256=sha(checkpoint),
                training_manifest_sha256=binding['manifest']['sha256'],finite_checkpoint=tensor_audit,
                iteration=499,common_step_counter=12000,last_transition_index=11999)

def slot_attempts(scope):
    if scope not in ('own_cpu32','science160'):raise ValueError('undeclared scope')
    return 32 if scope=='own_cpu32' else 160

def validate_matrix(manifest):
    policies=manifest.get('policies',[]);slots=manifest.get('slots',[])
    if manifest.get('campaign_root')!=CAMPAIGN_ROOT or manifest.get('claim_root')!=CLAIM_ROOT:
        raise ValueError('fixed new evaluation allocation roots required')
    expected={(seed,arm) for seed in SEEDS for arm in ARMS}
    if len(policies)!=6 or {(p['training_seed'],p['arm']) for p in policies}!=expected:raise ValueError('exact six training policies required')
    ids=set()
    for policy in policies:
        if (policy['policy_id']!=f"{policy['arm']}-seed{policy['training_seed']}" or policy['task']!=BASE+'-Horizon-'+policy['arm']
            or policy['native_window_s']!={'native50':.05,'extended200':.2}[policy['arm']]):raise ValueError('policy task identity mismatch')
        ids.add(policy['policy_id'])
    if len(slots)!=18 or hashlib.sha256(json.dumps(slots[0],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=HISTORICAL_SLOT0_SHA256:
        raise ValueError('historical qualified slot0 must remain byte-identical and non-issuable')
    actual=[]
    for index,slot in enumerate(slots):
        scope=slot['intent']['scope'];bank=slot['intent']['evaluation_seed'];policy=slot['intent']['policy']
        if policy not in policies or slot['intent']['attempts']!=slot_attempts(scope):raise ValueError('slot policy/population mismatch')
        if type(bank)is not int or (scope=='science160' and bank not in BANKS):raise ValueError('undeclared bank')
        if slot['intent'].get('cases')!=(['original-route+1'] if scope=='own_cpu32' else CASES):
            raise ValueError('exact physical start cases required')
        if slot['intent'].get('retry_allowed')is not False or slot['intent'].get('resume_allowed')is not False:
            raise ValueError('no retry/resume')
        tag='runtime-budget007' if index==3 else 'staged-integration002'
        if index==3 and (policy['policy_id'],scope,bank)!=('extended200-seed2026102301','own_cpu32',2026101602):
            raise ValueError('replacement belongs only to exact index3 own32 policy')
        if index and (type(slot['limits'].get('wall_seconds'))is not int or slot['limits']['wall_seconds']!={'own_cpu32':7200,'science160':36000}[scope]):
            raise ValueError('full20s own7200/five-case science36000 wall budget required')
        suffix='own_cpu32'if scope=='own_cpu32'else f'science160-bank{bank}'
        if scope=='own_cpu32':
            if bank!=2026101602:raise ValueError('exact technical bank required')
            suffix+='-'+tag
        key=policy['policy_id']+'-'+('own_cpu32'if scope=='own_cpu32'else f'science160-bank{bank}')+'-'+tag
        if slot['intent'].get('claim_key')!=key or slot['intent'].get('claim_path')!=str(Path(CLAIM_ROOT)/(key+'.json')):
            raise ValueError('exact finite fresh claim required')
        if (slot['intent']['output_root']!=str(Path(CAMPAIGN_ROOT)/'runs'/policy['policy_id']/suffix)
            or slot['private_root']!=str(Path(CAMPAIGN_ROOT)/'private'/policy['policy_id']/suffix)):
            raise ValueError('fixed per-policy output/private allocation required')
        actual.append((policy['policy_id'],scope,bank if scope=='science160' else None))
    desired={(name,scope,bank) for name in ids for scope,bank in [('own_cpu32',None),('science160',BANKS[0]),('science160',BANKS[1])]}
    if len(slots)!=18 or set(actual)!=desired or len(set(actual))!=18:raise ValueError('exact six technical and twelve scientific slots required')
    return True


def pinned_source(reference):
    path=Path(reference['path'])
    if not path.is_absolute() or any(parent.is_symlink() for parent in (path,*path.parents)):
        raise ValueError('exact nonsymlink pinned source required')
    source=path.read_bytes()
    if len(source)>131072 or hashlib.sha256(source).hexdigest()!=reference['sha256']:
        raise ValueError('legacy source bytes changed')
    return source


def load_legacy_consumer():
    source=pinned_source(LEGACY_SUPERVISOR)
    module=types.ModuleType('legacy_565_first_qualification');module.__file__=LEGACY_SUPERVISOR['path']
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module


def expected_worker_sha256():
    # Exact deterministic admission-only delta avoids a supervisor/worker hash cycle.
    source=pinned_source(LEGACY_WORKER).decode()
    old_block="    if scope=='science160':\n        seal=json.loads(Path(intent['own_cpu32_seal_path']).read_text())\n        if digest(intent['own_cpu32_seal_path'])!=intent['own_cpu32_seal_sha256']:\n            raise ValueError('own CPU32 seal changed')\n        for key in ('policy','runtime_identity','source_identity','observer_identity'):\n            if seal[key]!=intent[key]:raise ValueError('own CPU32 identity differs: '+key)\n        if seal['status']!='QUALIFIED' or seal['attempts']!=32:\n            raise ValueError('own CPU32 incomplete')\n"
    new_check="    if intent['scope']=='science160':\n        manifest=module.load_release(context['release_path'])\n        reference=manifest['slots'][context['slot']]['qualification_seal']\n        if reference!={'path':intent['own_cpu32_seal_path'],'sha256':intent['own_cpu32_seal_sha256']}:\n            raise ValueError('own CPU32 seal reference differs from admitted slot')\n        module.verify_qualification(intent,reference)  # Recompute from pinned source before the physical claim.\n"
    if source.count(old_block)!=1 or source.count('    return result\n')!=2:
        raise ValueError('legacy worker transformation differs')
    old_loader="    if supervisor!=trusted or context.get('supervisor_sha256')!=TRUSTED_SUPERVISOR_SHA256 or digest(supervisor)!=TRUSTED_SUPERVISOR_SHA256:\n        raise ValueError('trusted supervisor bytes mismatch')\n    spec=importlib.util.spec_from_file_location('trusted_cpu_supervisor_guard',supervisor)\n    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n"
    new_loader="    if supervisor!=trusted or context.get('supervisor_sha256')!=TRUSTED_SUPERVISOR_SHA256:\n        raise ValueError('trusted supervisor path/context mismatch')\n    source=supervisor.read_bytes()\n    if hashlib.sha256(source).hexdigest()!=TRUSTED_SUPERVISOR_SHA256:\n        raise ValueError('trusted supervisor bytes mismatch')\n    module=types.ModuleType('trusted_cpu_supervisor_guard');module.__file__=str(supervisor)\n    exec(compile(source,str(supervisor),'exec'),module.__dict__)  # Execute exactly the buffer whose SHA was checked; never cached bytecode.\n"
    if source.count(old_loader)!=1 or source.count('import time\n')!=1:
        raise ValueError('legacy worker source-compilation transformation differs')
    source=source.replace('import time\n','import time\nimport types\n',1).replace(old_loader,new_loader,1)
    source=source.replace(LEGACY_SUPERVISOR['sha256'],sha(__file__))
    source=source.replace(old_block,'').replace('    return result\n',new_check+'    return result\n',1)
    return hashlib.sha256(source.encode()).hexdigest()


def current_control_hashes(intent,old_observer):
    if Path(__file__).resolve()!=CURRENT_PACKAGE/'supervisor/cpu_supervisor.py':
        raise ValueError('exact runtime-budget007 primary source path required')
    expected={'cpu_supervisor.py':(CURRENT_PACKAGE/'supervisor/cpu_supervisor.py',sha(__file__)),
        'horizon_worker.py':(CURRENT_PACKAGE/'horizon_worker.py',expected_worker_sha256()),
        'horizon_integration.py':(CURRENT_PACKAGE/'horizon_integration.py',DIAGNOSTIC_INTEGRATION_SHA256)}
    rows={Path(row['path']).name:row for row in intent['package_files']}
    if len(rows)!=len(intent['package_files']):raise ValueError('ambiguous compatibility package roles')
    controls={name:digest for name,(path,digest)in expected.items()}
    identity=dict(old_observer,**controls)
    exact={name:{'path':str(CURRENT_PACKAGE/('supervisor/cpu_supervisor.py' if name=='cpu_supervisor.py' else name)),
        'sha256':digest}for name,digest in identity.items()}
    if rows!=exact:raise ValueError('exact ten-file007 producer paths/hashes required')
    for row in exact.values():pinned_source(row)
    return controls


def verify_legacy_first(intent,reference):
    if reference!=LEGACY_FIRST_SEAL:raise ValueError('only exact retained first qualification seal is compatible')
    original=bound(LEGACY_FIRST_INTENT);seal=bound(reference)
    if seal.get('independent_backup')!=LEGACY_FIRST_BACKUP:
        raise ValueError('exact retained first full-backup reference required')
    # Always execute the old actual consumer, including when identities equal old0.
    # Its original raw/prolog/backup checks remain authoritative for historical data.
    legacy=load_legacy_consumer()
    verified=legacy.verify_qualification(original,reference)
    for key in ('policy','runtime_identity','source_identity'):
        if intent.get(key)!=original.get(key):raise ValueError('legacy qualification non-observer identity drift')
    expected=original['observer_identity']
    if intent.get('scope')=='science160':expected=dict(expected,**current_control_hashes(intent,expected))
    if intent.get('observer_identity')!=expected:
        raise ValueError('legacy qualification requires exact old or three-producer007 observer map')
    return verified

def verify_qualification(intent,reference):
    if intent.get('policy',{}).get('policy_id')=='native50-seed2026102301':
        return verify_legacy_first(intent,reference)
    seal=bound(reference)
    if seal.get('status')!='QUALIFIED' or seal.get('attempts')!=32:raise ValueError('own actual CPU32 qualification required')
    for key in ('policy','runtime_identity','source_identity','observer_identity'):
        if seal.get(key)!=intent.get(key):raise ValueError('own qualification identity mismatch')
    if seal.get('raw_review')!='PASS' or seal.get('runtime_review')!='PASS' or seal.get('parity_review')!='PASS':
        raise ValueError('own actual raw/runtime/parity review required')
    raw_root=Path(seal.get('raw_root',''))
    members=inventory(raw_root)
    if tree_digest(members)!=seal.get('raw_tree_sha256'):raise ValueError('own CPU32 raw bytes changed')
    terminal=bound(seal.get('supervisor_terminal'))
    if terminal.get('status')!='RAW_REVIEW_REQUIRED' or terminal.get('returncode')!=0:
        raise ValueError('own CPU32 actual terminal success required')
    hardware_reference=seal.get('hardware_prolog')
    hardware=bound(hardware_reference)
    if Path(hardware_reference['path']).resolve()!=raw_root.resolve()/'hardware-prolog.json':raise ValueError('own CPU32 prolog path mismatch')
    validate_hardware_prolog(hardware_reference['path'],intent=intent)
    result=json.loads((raw_root/'worker-result.json').read_text())
    if (result.get('attempts')!=32 or result.get('checkpoint_contract_validated_before_overlay')is not True
        or result.get('final_contract_restored')is not True or result.get('reset_frames_sampled')!=0
        or result.get('hardware_prolog_sha256')!=hardware_reference['sha256']):
        raise ValueError('own CPU32 actual worker evidence missing')
    backup=bound(seal.get('independent_backup'))
    if backup.get('members')!=members or backup.get('tree_sha256')!=tree_digest(members):
        raise ValueError('own CPU32 complete archive member map required')
    if backup.get('status')!='PASS' or backup.get('independent_readback')!='PASS' or backup.get('policy')!=intent['policy']:
        raise ValueError('own CPU32 complete independent backup required')
    return seal

def verify_files(rows):
    if not rows:raise ValueError('empty required identity map')
    for row in rows:
        path=Path(row['path'])
        if not path.is_absolute() or path.is_symlink() or sha(path)!=row['sha256']:raise ValueError('sealed bytes changed')

RUNTIME_MODULES={'torch':'torch','mujoco':'mujoco','mujoco_warp':'mujoco_warp','mjlab':'mjlab','rsl_rl':'rsl_rl',
                 'mjlab_envs':'mjlab.envs','mjlab_env':'mjlab.envs.manager_based_rl_env',
                 'mjlab_runner':'mjlab.rl.runner','rsl_runner':'rsl_rl.runners.on_policy_runner'}
RUNTIME_SUFFIXES={role:name.replace('.','/')+('/__init__.py'if role in('torch','mujoco','mujoco_warp','mjlab','rsl_rl','mjlab_envs')else'.py')
                  for role,name in RUNTIME_MODULES.items()}


def verify_runtime_module(intent,role,module):
    expected=Path(intent['runtime_roles'][role]).resolve(strict=True)
    actual=getattr(module,'__file__',None)
    hashes={row['path']:row['sha256']for row in intent['runtime_files']}
    declared=intent['runtime_roles'][role]
    if actual is None or Path(actual).resolve()!=expected or declared not in hashes or sha(expected)!=hashes[declared]:
        raise ValueError('actual imported runtime origin/hash mismatch: '+role)
    return dict(path=str(expected),sha256=sha(expected))


def verify_imported_runtime(intent):
    origins={};modules={}
    for role,name in RUNTIME_MODULES.items():
        module=importlib.import_module(name)
        origins[role]=verify_runtime_module(intent,role,module);modules[role]=module
    selected=modules['mjlab_envs'].ManagerBasedRlEnv
    if selected is not modules['mjlab_env'].ManagerBasedRlEnv or Path(inspect.getfile(selected)).resolve()!=Path(intent['runtime_roles']['mjlab_env']).resolve():
        raise ValueError('actual environment class alias/implementation escaped pinned runtime')
    return origins


def expected_runtime_origins(intent):
    hashes={row['path']:row['sha256']for row in intent['runtime_files']}
    return {role:dict(path=str(Path(path).resolve()),sha256=hashes[path])for role,path in intent['runtime_roles'].items()}


def verify_intent_inventory(intent):
    for group in ('source_files','asset_files','package_files','training_evidence','runtime_files'):
        verify_files(intent[group])
    source=Path(intent['source_root']).resolve()
    actual=inventory(source)
    if any('__pycache__' in Path(name).parts or Path(name).suffix in ('.pyc','.pyo') for name in actual):
        raise ValueError('frozen source bytecode forbidden')
    declared={str(Path(row['path']).resolve().relative_to(source)):row['sha256'] for row in intent['source_files']}
    if actual!=declared:raise ValueError('complete frozen source member set required')
    critical={'scripts/evaluate_impact_speed.py','src/tasks/hammer/config/z1/spline_horizon_cfg.py',
              'src/tasks/hammer/mdp/spline_horizon.py','src/tasks/hammer/cat/hook.py','src/tasks/hammer/rl/runner.py'}
    if not critical.issubset(actual):raise ValueError('critical current source missing')
    roles=intent.get('runtime_roles',{})
    suffixes=RUNTIME_SUFFIXES
    runtime={row['path']:row['sha256'] for row in intent['runtime_files']}
    if set(roles)!=set(suffixes):raise ValueError('named runtime roles required')
    if any(not roles[key].endswith('/'+suffix) or roles[key] not in runtime for key,suffix in suffixes.items()):
        raise ValueError('runtime role binding mismatch')
    assets={row['path']:row['sha256'] for row in intent['asset_files']}
    asset_roles=intent.get('asset_roles',{})
    if set(asset_roles)!={'scene_xml','robot_xml'} or any(path not in assets or not path.endswith('.xml') for path in asset_roles.values()):
        raise ValueError('scene/robot asset roles required')
    python=intent.get('python',{})
    if Path(sys.executable).resolve()!=Path(python.get('path','')) or sha(sys.executable)!=python.get('sha256'):
        raise ValueError('interpreter identity mismatch')


def allocation_contract_sha256(manifest):
    contract={key:value for key,value in manifest.items()if key not in('authority','authority_archive')}
    return hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_authority(manifest):
    authority=bound(manifest.get('authority'));archive=bound(manifest.get('authority_archive'))
    if (authority.get('status')!='ADMITTED_ONCE' or authority.get('supervisor_sha256')!=manifest['supervisor_sha256']
        or authority.get('slot_keys')!=[slot['intent']['claim_key'] for slot in manifest['slots']]
        or authority.get('allocation_contract_sha256')!=allocation_contract_sha256(manifest)
        or archive.get('status')!='PASS' or archive.get('independent_readback')!='PASS'
        or archive.get('authority_sha256')!=manifest['authority']['sha256']):
        raise ValueError('independently preserved exact finite allocation authority required')


def load_release(path):
    release=json.loads(Path(path).read_text());manifest=bound(release.get('manifest'))
    if release.get('status')!='ADMITTED_ONCE' or manifest.get('schema')!='z1_horizon_cpu_supervisor_v1':raise ValueError('actual finite release required')
    if manifest.get('supervisor_sha256')!=sha(__file__):raise ValueError('supervisor bytes changed')
    validate_matrix(manifest)
    verify_authority(manifest)
    return manifest

def require_selected(manifest,index):
    verify_authority(manifest)
    authority=bound(manifest.get('authority'));selected=authority.get('selected_indices');phase=authority.get('execution_phase')
    if (not isinstance(selected,list) or not selected
        or any(type(i)is not int or not 0<=i<18 for i in selected)
        or selected!=sorted(set(selected))):
        raise ValueError('exact execution phase selected indices required')
    expected=(selected if phase=='remaining_own32' and selected in ([3],[6,9,12,15])
              else selected if phase=='science160' and all(i in (1,2,4,5,7,8,10,11,13,14,16,17) for i in selected) else None)
    if expected is None or selected!=expected:
        raise ValueError('exact execution phase selected indices required')
    scope='science160' if phase=='science160' else 'own_cpu32'
    if any(manifest['slots'][i]['intent'].get('scope')!=scope for i in selected):
        raise ValueError('selected intent scope differs from execution phase')
    if (authority.get('status')!='ADMITTED_ONCE' or type(index)is not int or index not in selected
        or any(manifest['slots'][i]['intent'].get('status')!='ADMITTED_ONCE' for i in selected)):
        raise ValueError('index or selected intents are not admitted by exact phase authority')
    if phase=='remaining_own32':
        first=manifest['slots'][0]['intent']
        if first.get('scope')!='own_cpu32':
            raise ValueError('remaining phase requires own32 in the first slot')
        seal=verify_qualification(first,authority.get('first_qualification_seal'))
        if Path(seal['raw_root']).resolve()!=Path(first['output_root']).resolve():
            raise ValueError('first qualification must bind the actual first slot raw root')
        if selected==[6,9,12,15]:
            replacement=manifest['slots'][3]['intent']
            seal=verify_qualification(replacement,authority.get('replacement_qualification_seal'))
            if Path(seal['raw_root']).resolve()!=Path(replacement['output_root']).resolve():
                raise ValueError('replacement qualification must bind actual fresh index3 raw root')


def headroom(root):return int(os.getxattr(root,'ceph.quota.max_bytes'))-int(os.getxattr(root,'ceph.dir.rbytes'))

def directory_bytes(root):
    # Live caches remove entries during scans. Ignore only observed disappearance;
    # all other filesystem errors remain fatal, and never follow observed links.
    def scan_error(error):
        if not isinstance(error,FileNotFoundError):raise error
    def entry_stat(path):
        try:result=os.lstat(path)
        except FileNotFoundError:return None
        if stat.S_ISLNK(result.st_mode):raise ValueError('output/private symlink')
        return result
    root_stat=entry_stat(root)
    if root_stat is None:return 0
    if not stat.S_ISDIR(root_stat.st_mode):raise ValueError('output/private directory required')
    total=0
    for directory,dirs,files in os.walk(root,topdown=True,onerror=scan_error,followlinks=False):
        for name in list(dirs):
            result=entry_stat(os.path.join(directory,name))
            if result is None or not stat.S_ISDIR(result.st_mode):dirs.remove(name)
        for name in files:
            result=entry_stat(os.path.join(directory,name))
            if result is not None and stat.S_ISREG(result.st_mode):total+=result.st_size
    return total

def process_identity(pid):
    # Linux stat field22 is creation clock; comm may itself contain spaces.
    fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
    return dict(pid=pid,pgid=int(fields[2]),session=int(fields[3]),start_ticks=int(fields[19]))


def owned_members(ownership):
    if not isinstance(ownership,dict)or set(ownership)!={'pid','pgid','session','start_ticks'}:
        raise ValueError('captured owned process/session identity required')
    pid=ownership['pid']
    if pid<=1 or ownership['pgid']!=pid or ownership['session']!=pid or pid==os.getpgrp():
        raise ValueError('refuse to signal unrelated process group')
    try:
        if process_identity(pid)!=ownership:raise ValueError('leader PID/session/start identity changed')
    except (FileNotFoundError,ProcessLookupError):pass
    members=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:actual=process_identity(int(path.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if actual['pgid']!=pid:continue
        if actual['session']!=pid or actual['start_ticks']<ownership['start_ticks']:
            raise ValueError('group/session ownership changed')
        members.append(actual['pid'])
    return members


def stop(process,ownership):
    if not isinstance(ownership,dict)or ownership.get('pid')!=process.pid:raise ValueError('no matching captured isolated process group identity')
    process.poll()
    if owned_members(ownership):
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        deadline=time.monotonic()+10
        while True:
            process.poll()
            if not owned_members(ownership):break
            if time.monotonic()>=deadline:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                kill_deadline=time.monotonic()+2
                while True:
                    process.poll()
                    if not owned_members(ownership):break
                    if time.monotonic()>=kill_deadline:raise RuntimeError('owned descendants survived bounded cleanup')
                    time.sleep(.05)
                break
            time.sleep(.05)
    if process.poll()is None:process.wait(timeout=1)


def cleanup_result(process,ownership):
    if process is None:return None
    try:stop(process,ownership)
    except BaseException as error:return repr(error)
    return None


def record_terminal(output,private,process,identity,status,reason,cleanup_error):
    sizes={};measurement_errors={}
    for key,path in(('output_bytes',output),('private_bytes',private)):
        try:sizes[key]=directory_bytes(path)
        except BaseException as error:sizes[key]=None;measurement_errors[key]=repr(error)
    durable(Path(output)/'supervisor-terminal.json',dict(status=status,reason=reason,
        returncode=None if process is None else process.returncode,allocation=identity,
        cleanup_error=cleanup_error,measurement_errors=measurement_errors,retry_allowed=False,**sizes))

def allocation():
    host=os.uname().nodename
    if (host.split('.')[0]!='cn0724' or not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='2'
        or os.environ.get('SLURM_JOB_NODELIST')!='cn0724' or os.environ.get('SLURM_MEM_PER_NODE')!='8192'
        or os.environ.get('CUDA_VISIBLE_DEVICES') or len(os.sched_getaffinity(0))!=2
        or any(os.environ.get(key)!='2'for key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))):
        raise ValueError('actual cn0724 CPU2 affinity/threads/mem8192 allocation required')
    return dict(hostname=host,**{key:os.environ[key] for key in ('SLURM_JOB_ID','SLURM_CPUS_PER_TASK','SLURM_JOB_NODELIST','SLURM_MEM_PER_NODE')})


def hardware_prolog(intent_path):
    # Called once by the supervised child, before guard auditing or any cases.
    intent=json.loads(Path(intent_path).read_text())
    import torch
    verify_runtime_module(intent,'torch',torch)  # Before any Torch thread/tensor operation.
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    origins=verify_imported_runtime(intent)
    data=dict(status='PASS',allocation=allocation(),affinity=sorted(os.sched_getaffinity(0)),
              torch_intraop=torch.get_num_threads(),torch_interop=torch.get_num_interop_threads(),
              thread_environment={key:os.environ[key]for key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
              pid=os.getpid(),before_environment_construction=True,runtime_origins=origins)
    if data['torch_intraop']!=2 or data['torch_interop']!=2:raise ValueError('actual Torch thread configuration failed')
    path=Path(intent_path).parent/'hardware-prolog.json';durable(path,data)
    return path


def validate_hardware_prolog(path,*,live=False,intent=None):
    data=json.loads(Path(path).read_text())
    if (data.get('status')!='PASS'or data.get('before_environment_construction')is not True
        or data.get('allocation',{}).get('hostname','').split('.')[0]!='cn0724'
        or not data.get('allocation',{}).get('SLURM_JOB_ID')
        or data.get('allocation',{}).get('SLURM_CPUS_PER_TASK')!='2'
        or data.get('allocation',{}).get('SLURM_JOB_NODELIST')!='cn0724'
        or data.get('allocation',{}).get('SLURM_MEM_PER_NODE')!='8192'
        or len(data.get('affinity',[]))!=2 or data.get('torch_intraop')!=2 or data.get('torch_interop')!=2
        or data.get('thread_environment')!={key:'2'for key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}):
        raise ValueError('actual CPU host/affinity/thread prolog required')
    if intent is not None and data.get('runtime_origins')!=expected_runtime_origins(intent):
        raise ValueError('sealed actual runtime import origins differ')
    if live:
        if intent is None:raise ValueError('live runtime origins require exact intent')
        import torch
        verify_runtime_module(intent,'torch',torch)
        if verify_imported_runtime(intent)!=data['runtime_origins']:raise ValueError('live runtime import origins changed')
        if (data['allocation']!=allocation()or data['affinity']!=sorted(os.sched_getaffinity(0))or data.get('pid')!=os.getpid()
            or torch.get_num_threads()!=2 or torch.get_num_interop_threads()!=2):raise ValueError('live hardware prolog changed')
    return data

def supervise(release_path,index):
    manifest=load_release(release_path)
    require_selected(manifest,index)
    if type(index)is not int or not 0<=index<18:raise ValueError('slot outside finite allocation')
    slot=manifest['slots'][index];intent=dict(slot['intent']);identity=allocation()
    verify_intent_inventory(intent)
    packages={Path(row['path']).name:row for row in intent['package_files']}
    package_root=Path(packages.get('horizon_worker.py',{}).get('path','')).parent
    for path in package_root.rglob('*'):
        if path.is_symlink()or '__pycache__'in path.parts or path.suffix in('.pyc','.pyo'):
            raise ValueError('frozen evaluator package cache/symlink forbidden')
        if path.is_file()and path.suffix=='.py'and str(path)not in {row['path']for row in intent['package_files']}:
            raise ValueError('unmanifested evaluator Python producer')
    if len(packages)!=len(intent['package_files']):raise ValueError('ambiguous package basename roles')
    for name in ('horizon_worker.py','horizon_integration.py','horizon_eval_preflight.py','safety_observer.py','continuation_tools.py','reset_pose.py','current_evaluator_candidate.py','onset_trace.py','current-evaluator-origin.json'):
        if name not in packages:raise ValueError('missing evaluator/observer package role')
    if (intent['reviewed_hook_path']!=packages['horizon_integration.py']['path']
        or intent['reviewed_hook_sha256']!=packages['horizon_integration.py']['sha256']
        or Path(intent['reviewed_hook_path']).parent!=Path(packages['horizon_worker.py']['path']).parent):
        raise ValueError('worker/hook package role mismatch')
    observer={name:row['sha256'] for name,row in packages.items()}
    if intent.get('observer_identity')!=observer:raise ValueError('exact observer/evaluator identity required')
    source=Path(intent['source_root'])
    if any(p.is_symlink() or '__pycache__' in p.parts or p.suffix in ('.pyc','.pyo') for p in source.rglob('*')):raise ValueError('source caches/symlinks forbidden')
    audit=audit_training(slot['training_bindings'],tensor_audit=False)
    training_manifest=bound(slot['training_bindings']['manifest'])
    source_map={str(Path(row['path']).resolve().relative_to(source.resolve())):row['sha256'] for row in intent['source_files']}
    if source_map!=training_manifest.get('source_files'):raise ValueError('evaluation source differs from own frozen training source')
    for group in ('runtime_files','asset_files'):
        if {row['path']:row['sha256'] for row in intent[group]}!=training_manifest.get(group):
            raise ValueError('evaluation runtime/assets differ from training manifest')
    expected_source=dict(training_manifest_sha256=slot['training_bindings']['manifest']['sha256'],source_tree_sha256=tree_digest(source_map))
    if intent.get('source_identity')!=expected_source:raise ValueError('exact source identity required')
    if audit['policy']!=intent['policy']:raise ValueError('training audit policy mismatch')
    if intent['scope']=='science160':
        verify_qualification(intent,slot['qualification_seal'])
        intent['own_cpu32_seal_path']=slot['qualification_seal']['path'];intent['own_cpu32_seal_sha256']=slot['qualification_seal']['sha256']
    cap=slot['limits'];floor=(512+32)*1024**2
    if any(type(cap.get(key))is not int or cap[key]<=0 for key in ('wall_seconds','output_bytes','private_bytes','future_reserve_bytes')):raise ValueError('admitted exact positive limits required')
    if headroom(manifest['quota_root'])<floor+cap['future_reserve_bytes']:raise ValueError('future output capacity not admitted')
    output=Path(intent['output_root']);claim=Path(intent['claim_path'])
    private=Path(slot['private_root'])
    if (not output.is_absolute() or not private.is_absolute() or output==private or output.is_relative_to(private)
        or private.is_relative_to(output) or output.is_symlink() or private.is_symlink()):
        raise ValueError('distinct absolute new output/private directories required')
    ledger=Path(manifest['claim_root'])
    if not ledger.is_absolute() or ledger.is_symlink():raise ValueError('absolute unchanged ledger required')
    if claim.parent!=ledger or claim.name!=intent['claim_key']+'.json':raise ValueError('cohort claim namespace mismatch')
    output.mkdir(exist_ok=False);private.mkdir(exist_ok=False)
    intent.update(deadline_epoch=time.time()+cap['wall_seconds'],max_output_bytes=cap['output_bytes'],
        supervisor_context=dict(parent=os.getpid(),release_path=str(Path(release_path).resolve()),
            release_sha256=sha(release_path),slot=index,worker_sha256=packages['horizon_worker.py']['sha256'],
            supervisor_path=str(Path(__file__).resolve()),supervisor_sha256=sha(__file__)))
    intent_path=output/'intent.json';durable(intent_path,intent);durable(output/'training-metadata-preflight.json',audit)
    durable(ledger/(intent['claim_key']+'.supervisor.json'),dict(slot=index,parent=os.getpid(),allocation=identity,
        release_sha256=sha(release_path),intent_sha256=sha(intent_path),retry_allowed=False))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for key in ('TMPDIR','XDG_CACHE_HOME','WARP_CACHE_PATH','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR'):
        path=private/key.lower();path.mkdir();env[key]=str(path)
    process=None;reason=None;ownership=None;cleanup_error=None;start=time.monotonic()
    try:
        with (output/'worker.log').open('xb') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--child-intent',str(intent_path),
                '--parent',str(os.getpid()),'--worker',packages['horizon_worker.py']['path']],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
            ownership=process_identity(process.pid)
            if ownership['pgid']!=process.pid or ownership['session']!=process.pid:raise ValueError('actual isolated child ownership failed')
            durable(output/'process-group-ownership.json',dict(identity=ownership,parent=os.getpid(),intent_sha256=sha(intent_path)))
            while process.poll()is None:
                if time.monotonic()-start>=cap['wall_seconds']:reason='WALL_LIMIT'
                elif directory_bytes(output)>=cap['output_bytes']:reason='OUTPUT_LIMIT'
                elif directory_bytes(private)>=cap['private_bytes']:reason='PRIVATE_LIMIT'
                elif headroom(manifest['quota_root'])<floor:reason='PRESERVATION_FLOOR'
                if reason:stop(process,ownership);break
                time.sleep(1)
        stop(process,ownership)
        if (directory_bytes(output)>=cap['output_bytes']or directory_bytes(private)>=cap['private_bytes']
            or headroom(manifest['quota_root'])<floor):raise ValueError('final output/private/floor guard failed')
        result_path=output/'worker-result.json'
        status='RAW_REVIEW_REQUIRED' if reason is None and process.returncode==0 and result_path.is_file() else 'FAILED_NO_RETRY'
    except BaseException as error:
        cleanup_error=cleanup_result(process,ownership)
        status='FAILED_NO_RETRY';reason=repr(error)
    record_terminal(output,private,process,identity,status,reason,cleanup_error)
    if status!='RAW_REVIEW_REQUIRED':raise SystemExit(1)

def parent_command(parent):
    command=(Path('/proc')/str(parent)/'cmdline').read_bytes().split(b'\0')
    return [part.decode() for part in command if part]


def authorize_physical_worker(intent_path,worker_path):
    """Called inside physical worker before its claim or integration import."""
    intent_path=Path(intent_path).resolve(strict=True)
    intent=json.loads(intent_path.read_text());context=intent.get('supervisor_context')
    if not isinstance(context,dict):raise ValueError('physical worker requires live supervisor context')
    parent=context.get('parent')
    if type(parent)is not int or parent!=os.getppid():raise ValueError('physical worker parent mismatch')
    supervisor=Path(context['supervisor_path']).resolve(strict=True)
    if supervisor!=Path(__file__).resolve() or sha(supervisor)!=context['supervisor_sha256']:
        raise ValueError('trusted supervisor source identity mismatch')
    release_path=Path(context['release_path']).resolve(strict=True)
    if sha(release_path)!=context['release_sha256']:raise ValueError('actual supervisor release changed')
    manifest=load_release(release_path)
    index=context['slot']
    if type(index)is not int or not 0<=index<18:raise ValueError('exact slot required')
    require_selected(manifest,index)
    slot=manifest['slots'][index]
    if (manifest['supervisor_sha256']!=context['supervisor_sha256'] or sha(worker_path)!=context['worker_sha256']
        or any(intent.get(key)!=value for key,value in slot['intent'].items())
        or intent['max_output_bytes']!=slot['limits']['output_bytes']
        or not time.time()<intent['deadline_epoch']<=time.time()+slot['limits']['wall_seconds']):
        raise ValueError('actual admitted slot/worker/clock mismatch')
    command=parent_command(parent)
    expected_tail=[str(supervisor),'--release',str(release_path),'--index',str(index)]
    if len(command)!=6 or command[1:]!=expected_tail or Path(command[0]).resolve()!=Path(sys.executable).resolve():
        raise ValueError('parent is not the trusted bounded supervisor invocation')
    ledger=Path(manifest['claim_root'])
    record=json.loads((ledger/(intent['claim_key']+'.supervisor.json')).read_text())
    if (record.get('parent')!=parent or record.get('slot')!=index or record.get('intent_sha256')!=sha(intent_path)
        or record.get('release_sha256')!=context['release_sha256'] or record.get('allocation')!=allocation()):
        raise ValueError('live once claim/intent/allocation mismatch')
    group_record=json.loads((intent_path.parent/'process-group-ownership.json').read_text())
    if (group_record.get('identity')!=process_identity(os.getpid())or group_record.get('parent')!=parent
        or group_record.get('intent_sha256')!=sha(intent_path)):raise ValueError('actual isolated child identity changed')
    verify_intent_inventory(intent)
    prolog=Path(intent_path).parent/'hardware-prolog.json'
    validate_hardware_prolog(prolog,live=True,intent=intent)
    audit=audit_training(slot['training_bindings'],runtime_intent=intent)
    if audit['policy']!=intent['policy']or audit.get('status')!='PASS':raise ValueError('own actual training audit mismatch')
    durable(intent_path.parent/'training-audit.json',audit)
    if intent['scope']=='science160':verify_qualification(intent,slot['qualification_seal'])
    return dict(status='AUTHORIZED_LIVE_SUPERVISOR',slot=index,parent=parent,hardware_prolog_sha256=sha(prolog))


def child(intent_path,parent,worker):
    if sys.platform!='linux' or os.getppid()!=parent or ctypes.CDLL(None).prctl(1,signal.SIGTERM)!=0 or os.getppid()!=parent:
        raise ValueError('live supervisor parent-death protection required')
    intent=json.loads(Path(intent_path).read_text());context=intent['supervisor_context']
    if context['parent']!=parent or context['worker_sha256']!=sha(worker) or sha(context['release_path'])!=context['release_sha256']:
        raise ValueError('child authority/worker changed')
    manifest=load_release(context['release_path']);require_selected(manifest,context['slot']);slot=manifest['slots'][context['slot']]
    verify_intent_inventory(intent)
    ledger=Path(manifest['claim_root']);claim=json.loads((ledger/(intent['claim_key']+'.supervisor.json')).read_text())
    if (claim['parent']!=parent or claim['intent_sha256']!=sha(intent_path) or claim['allocation']!=allocation()
        or claim['release_sha256']!=context['release_sha256'] or slot['intent']['policy']!=intent['policy']):
        raise ValueError('child lacks live exact supervisor claim')
    ownership_path=Path(intent_path).parent/'process-group-ownership.json'
    ownership_deadline=time.monotonic()+2
    while True:
        try:owned=json.loads(ownership_path.read_text());break
        except(FileNotFoundError,json.JSONDecodeError):
            if time.monotonic()>=ownership_deadline:raise ValueError('supervisor ownership record missing/incomplete')
            time.sleep(.01)
    if (owned.get('identity')!=process_identity(os.getpid())or owned.get('parent')!=parent
        or owned.get('intent_sha256')!=sha(intent_path)):raise ValueError('child isolated group/start identity mismatch')
    hardware_prolog(intent_path)
    sys.path.insert(0,str(Path(worker).resolve().parent))
    spec=importlib.util.spec_from_file_location('bounded_horizon_worker',worker);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.run(intent_path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release');parser.add_argument('--index',type=int)
    parser.add_argument('--child-intent');parser.add_argument('--parent',type=int);parser.add_argument('--worker');args=parser.parse_args()
    if args.child_intent:child(args.child_intent,args.parent,args.worker)
    else:supervise(args.release,args.index)

if __name__=='__main__':main()
