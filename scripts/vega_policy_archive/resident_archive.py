#!/usr/bin/env python3
"""Vega-resident preservation; Mac independently verifies bytes without extraction.

Production entrypoints load a complete hash-bound source graph before importing
any helper. Only stdlib is imported by this bootstrap. No policy code is loaded.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import sys
import time
import types

MODE='VEGA_RESIDENT_FULL_ARCHIVE_MAC_STREAM_V1'
SCHEMA='z1_horizon_vega_resident_archive_request_v1'
CPU_ROOT='/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002'
REMOTE_PYTHON='/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python'
RESOLVED_PYTHON='/cephhome/eunikhilr/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'
PYTHON_SHA='d778d0b551287fb5f06c4464f78b616c086f8d42e94d8cfc209b12324cf937a1'
SUPERVISOR_SHA='565bb43027d8c9a98505603985d7a95b8f54c093c51e6cbcb2ba947f1507c4fd'
META=4194304
EVIDENCE=8388608
MIB=1024**2


def require(value,message):
    if not value:raise ValueError(message)


def check(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('resident archive deadline expired')


def json_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):return hashlib.sha256(json_bytes(value)).hexdigest()


def parse_json(data):
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,'duplicate JSON key');value[key]=item
        return value
    def invalid(value):raise ValueError('nonfinite JSON')
    return json.loads(data,object_pairs_hook=pairs,parse_constant=invalid)


def plain(value):
    path=Path(value)
    require(path.is_absolute() and str(path)==value and '..' not in path.parts,'canonical absolute path required')
    return path


def read_bytes(path,maximum,deadline):
    """No-follow bounded descriptor read, including parent directory checks."""
    path=plain(str(path))
    for parent in reversed(path.parents):require(stat.S_ISDIR(parent.lstat().st_mode),'nonsymlink parents required')
    check(deadline)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size<=maximum,'bounded regular file required')
        chunks=[];size=0
        while True:
            check(deadline);body=stream.read(min(MIB,maximum-size+1));size+=len(body)
            require(size<=maximum,'bounded file grew while reading')
            if not body:break
            chunks.append(body)
        after=os.fstat(stream.fileno())
        fields=('st_mode','st_dev','st_ino','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
        require(tuple(getattr(before,k) for k in fields)==tuple(getattr(after,k) for k in fields),'bound descriptor drift')
    return b''.join(chunks)


def bound_bytes(reference,deadline,maximum=EVIDENCE,base=None):
    require(isinstance(reference,dict) and set(reference)=={'path','sha256'},'exact path/SHA reference required')
    require(isinstance(reference['sha256'],str) and re.fullmatch('[0-9a-f]{64}',reference['sha256']),'SHA reference required')
    path=Path(reference['path'])
    if not path.is_absolute():
        require(base is not None and '..' not in path.parts,'bound relative path requires base');path=base/path
    body=read_bytes(path,maximum,deadline)
    require(hashlib.sha256(body).hexdigest()==reference['sha256'],'bound input SHA differs: '+str(path))
    return body


# This exact bootstrap is sent as the SSH -c argument. Its manifest hash is the
# caller's preservation-contract hash. No stage source executes before *all*
# source bytes and compiled code objects have passed the complete inventory.
BOOTSTRAP=r'''
import hashlib,json,os,stat,sys,types
from pathlib import Path

def load_verified_package(reference):
    def need(ok,msg):
        if not ok:raise ValueError(msg)
    def read(path,limit):
        need(path.is_absolute() and '..' not in path.parts,'absolute package path required')
        for parent in reversed(path.parents):need(stat.S_ISDIR(parent.lstat().st_mode),'package parent symlink forbidden')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as stream:
            before=os.fstat(stream.fileno());need(stat.S_ISREG(before.st_mode) and before.st_size<=limit,'bounded regular package source required')
            data=stream.read(limit+1);after=os.fstat(stream.fileno())
        fields=('st_mode','st_ino','st_dev','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
        need(len(data)<=limit and tuple(getattr(before,k) for k in fields)==tuple(getattr(after,k) for k in fields),'package descriptor drift')
        return data
    path=Path(reference['path']);data=read(path,131072)
    need(hashlib.sha256(data).hexdigest()==reference['sha256'],'package manifest SHA differs')
    def pairs(items):
        result={}
        for key,value in items:need(key not in result,'duplicate package key');result[key]=value
        return result
    manifest=json.loads(data,object_pairs_hook=pairs)
    need(set(manifest)=={'schema','entrypoint','files'} and manifest['schema']=='z1_vega_policy_archive_package_v1' and manifest['entrypoint']=='resident_archive.py','package manifest schema differs')
    expected={'__init__.py','compat/__init__.py','compat/archive_contract.py','compat/policy_archive.py','compat/remote_policy_archive.py','stream_reader.py','resident_store.py','capacity.py','transport.py','resident_archive.py'}
    need(set(manifest['files'])==expected,'package missing/extra/unbound source')
    root=path.parent;actual=set()
    def walk(directory):
        with os.scandir(directory) as entries:
            for entry in entries:
                relative=Path(entry.path).relative_to(root).as_posix()
                if relative=='tests':continue
                info=entry.stat(follow_symlinks=False)
                need(not stat.S_ISLNK(info.st_mode),'package symlink forbidden')
                if stat.S_ISDIR(info.st_mode):walk(Path(entry.path))
                elif relative.endswith('.py') or relative.endswith('.pyc'):actual.add(relative)
    walk(root);need(actual==expected,'package unexpected source/bytecode')
    sources={};codes={}
    for relative in sorted(expected):
        spec=manifest['files'][relative]
        need(set(spec)=={'sha256','bytes'} and type(spec['bytes'])is int and 0<spec['bytes']<=131072,'package file binding malformed')
        body=read(root/relative,131072)
        need(len(body)==spec['bytes'] and hashlib.sha256(body).hexdigest()==spec['sha256'],'package source SHA differs: '+relative)
        sources[relative]=body
        codes[relative]=compile(body,str(root/relative),'exec')
    # Legacy leaf sources are compiled above, then executed only by the independently
    # pinned compat loader. All package imports resolve already verified modules.
    names=[('__init__.py','vega_policy_archive'),('compat/__init__.py','vega_policy_archive.compat'),('stream_reader.py','vega_policy_archive.stream_reader'),('resident_store.py','vega_policy_archive.resident_store'),('capacity.py','vega_policy_archive.capacity'),('transport.py','vega_policy_archive.transport'),('resident_archive.py','vega_policy_archive.resident_archive')]
    result={}
    for relative,name in names:
        module=types.ModuleType(name);module.__file__=str(root/relative)
        module.__package__=name if relative.endswith('__init__.py') else name.rpartition('.')[0]
        if relative.endswith('__init__.py'):module.__path__=[str((root/relative).parent)]
        sys.modules[name]=module
        exec(codes[relative],module.__dict__);result[name]=module
    return result
'''


def verify_package(reference):
    namespace={};exec(compile(BOOTSTRAP,'<bound-package-bootstrap>','exec'),namespace)
    return namespace['load_verified_package'](reference)


def dependencies():
    from . import resident_store,capacity,transport,stream_reader
    from .compat import load_modules
    return dict(store=resident_store,capacity=capacity,transport=transport,reader=stream_reader,**load_modules())


def request_identity(request):
    return digest({k:v for k,v in request.items() if not k.startswith('_') and k!='capacity_ledger_reference'})


def remote_deadline(seconds):
    require(type(seconds) is int and 0<seconds<=560,'bounded remote wall seconds required')
    return time.monotonic()+seconds


def preservation(request,manifest):
    store=dependencies()['store']
    expected=dict(schema='z1_vega_resident_preservation_contract_v1',mode=MODE,
        backend_package=request['remote_helper_package_reference'],capacity_contract=request['remote_capacity_contract_reference'],
        backup_root=str(store.BACKUP_ROOT),mac_control_root=request['mac_control_root'])
    require(manifest.get('preservation_contract')==expected,'actual manifest preservation contract differs')
    require(request['backend_mode']==MODE and request['remote_archive_root']==str(store.BACKUP_ROOT),'closed backend mode/root required')
    for local,remote in [('helper_package_reference','remote_helper_package_reference'),('capacity_contract_reference','remote_capacity_contract_reference')]:
        require(request[local]['sha256']==request[remote]['sha256'],'local/remote helper/capacity SHA differs')
    return expected


def capacity_contract(request,contract,ledger):
    cap=dependencies()['capacity']
    require(contract==dict(schema='z1_vega_resident_capacity_contract_v1',rule_version=cap.RULE,
        slots=cap.layout_rows(),slot_caps=cap.CAPS,shared_caps=cap.SHARED,
        immutable_reserve_bytes=cap.IMMUTABLE,mac_envelope_bytes=cap.MAC_ENVELOPE,
        mac_retained_bytes=cap.MAC_RETAINED,mac_working_bytes=cap.MAC_WORKING),
        'capacity contract differs from fixed allocations')
    cap.remaining_budget(ledger)
    require(ledger['request_identity_sha256']==request_identity(request),'ledger request binding differs')
    require(ledger['release_sha256']==request['cpu_supervisor_release']['sha256'],'ledger release binding differs')
    require(ledger['layout_sha256']==request['cpu_layout_reference']['sha256'],'ledger layout binding differs')


def validate(request,request_path,deadline):
    """Retain unchanged CPU validator, then bind the resident-only contract."""
    require(request.get('schema')==SCHEMA,'resident archive request schema required')
    require(not any(key.startswith('_') for key in request),'caller internal fields forbidden')
    original=json.loads(json_bytes(request))
    legacy=json.loads(json_bytes(request));legacy['schema']='z1_horizon_terminal_multiroot_archive_request_v1'
    modules=dependencies();modules['policy'].validate(legacy,request_path,deadline)
    require(request['remote_python']==REMOTE_PYTHON,'approved lexical CPU interpreter required')
    require(request['wall_seconds']<=600,'hard wall bound exceeded');modules['store']._bounds(request)
    require(request['max_members']<=8192,'node cap exceeded')
    manifest=parse_json(bound_bytes(request['cpu_supervisor_manifest'],deadline,base=request_path.parent))
    preservation(request,manifest)
    mac_root=plain(request['mac_control_root'])
    for root in (plain(request['source_root']),plain(request['private_root'])):
        require(mac_root!=root and mac_root not in root.parents and root not in mac_root.parents,'Mac controls must be separate from source roots')
    package=bound_bytes(request['helper_package_reference'],deadline,base=request_path.parent)
    require(len(package)<=131072,'package manifest cap exceeded')
    contract=parse_json(bound_bytes(request['capacity_contract_reference'],deadline,base=request_path.parent))
    ledger=parse_json(bound_bytes(request['capacity_ledger_reference'],deadline,base=request_path.parent))
    capacity_contract(original,contract,ledger)
    require(request['remote_supervisor_release']['sha256']==request['cpu_supervisor_release']['sha256'],'remote release differs from local actual release')
    release=plain(request['remote_supervisor_release']['path'])
    require(release in {Path(CPU_ROOT)/'admission-staged-integration002'/phase/'release.json' for phase in ('first','remaining','science')},'actual closed phase admission release required')
    destination=modules['store'].validate_destination(request)
    require(request['remote_archive_destination']==str(destination),'derived remote destination differs')
    # Prevent post-validation identity changes, including previously omitted fields.
    request=original
    request.update({k:v for k,v in legacy.items() if k.startswith('_')})
    request['_request_identity_sha256']=request_identity(original)
    request['_ledger']=ledger;request['_capacity_contract']=contract
    return request


def normalized_receipt(request,verified,before,after,created,*,primary_reference,provenance_reference):
    origins=verified['origin_map']
    def members(prefix):return {name[len(prefix)+1:]:row['sha256'] for name,row in origins.items() if name.startswith(prefix+'/') and row['type']=='file'}
    raw=members('policy-tree');private=members('private-tree')
    return dict(schema='z1_horizon_terminal_multiroot_archive_receipt_v1',status='PASS',independent_readback='PASS',
        archive_only=True,readback_platform=sys.platform,readback_hostname=os.uname().nodename,
        technical_qualification=False,scientific_qualification=False,policy=request['policy'],scope=request['scope'],bank=request.get('bank'),
        run_root=request['source_root'],private_root=request['private_root'],members=raw,tree_sha256=digest(raw),
        private_members=private,private_tree_sha256=digest(private),training_manifest_sha256=request.get('training_manifest_sha256'),
        archive_path=created['archive_path'],archive_sha256=verified['archive_sha256'],archive_bytes=verified['archive_bytes'],
        job_before=before['job'],job_after=after['job'],shared_source=request['shared_source'],node_metadata=origins,raw_tree_extracted=False,
        backend_mode=MODE,remote_archive_location=created['destination'],primary_verification=primary_reference,
        independent_stream=dict(status='PASS',archive_sha256=verified['archive_sha256'],archive_bytes=verified['archive_bytes'],transport_exit=0,complete_eof=True),
        helper_package=request['helper_package_reference'],capacity_ledger=request['capacity_ledger_reference'],
        preservation_provenance=provenance_reference)


def control_read(transport,request,reference,deadline):
    result=transport.call('readcontrol',request,deadline=deadline,reference=reference)
    body=base64.b64decode(result['body_b64'],validate=True)
    require(len(body)<=META,'remote control byte cap exceeded')
    require(hashlib.sha256(body).hexdigest()==reference['sha256'],'remote control readback SHA differs')
    if 'bytes' in reference:require(len(body)==reference['bytes'],'remote control readback size differs')
    return parse_json(body)


def publish(transport,request,name,value,deadline):
    body=json_bytes(value);require(len(body)<=META,'published full metadata exceeds4MiB')
    result=transport.call('publish',request,deadline=deadline,name=name,body_b64=base64.b64encode(body).decode())
    expected=dict(path=request.get('remote_archive_destination',str(dependencies()['store'].validate_destination(request)))+'/'+name,sha256=hashlib.sha256(body).hexdigest(),bytes=len(body))
    require(result==expected,'remote publication binding differs')
    require(control_read(transport,request,result,deadline)==value,'published remote control differs')
    return result


def coordinate(request,transport,guard,*,deadline,recheck):
    """Serialize the complete operation on the admitted Mac control root."""
    with guard.lease():
        return _coordinate(request,transport,guard,deadline=deadline,recheck=recheck)


def _coordinate(request,transport,guard,*,deadline,recheck):
    """The only successful path; callback transport must guarantee exit zero."""
    modules=dependencies();store=modules['store'];destination=store.validate_destination(request)
    guard.check();check(deadline)
    created=transport.call('create',request,deadline=deadline)
    require(created.get('status')=='COMPLETE','remote archive did not complete')
    require(created.get('destination')==str(destination) and created.get('archive_path')==str(destination/'terminal-policy-full.tar.gz'),'remote archive location differs')
    require(type(created.get('archive_bytes')) is int and 0<created['archive_bytes']<=request['max_archive_bytes'] and re.fullmatch('[0-9a-f]{64}',created.get('archive_sha256','')),'complete archive byte binding required')
    refs=created.get('controls',{});require(set(refs)=={'attempt','before','after','primary','origin','index'},'complete individually bound controls required')
    names=dict(attempt='attempt.json',before='before.json',after='after.json',primary='verification.json',origin='origin.json',index='index.json')
    for key,ref in refs.items():require(ref.get('path')==str(destination/names[key]),'foreign create control path')
    controls={key:control_read(transport,request,refs[key],deadline) for key in names}
    before=controls['before'];after=controls['after'];primary=controls['primary'];index=controls['index']
    for inv in (before,after):
        for key in ('policy_id','scope','source_root','private_root'):require(inv.get(key)==request[key],'source inventory request identity differs: '+key)
        require(inv.get('job',{}).get('job_id')==request['job_id'],'source job differs')
        modules['remote'].validate_guard(request,inv['job'])
    store.compare_source_inventories(before,after)
    require(index.get('status')=='ARCHIVE_BOUND' and index.get('policy_id')==request['policy_id'] and index.get('job_id')==request['job_id'],'bound index identity differs')
    require(index.get('controls')=={k:v for k,v in refs.items() if k!='index'},'bound index control closure differs')
    for value in (primary,index):
        require(value.get('archive_sha256')==created['archive_sha256'] and value.get('archive_bytes')==created['archive_bytes'],'primary/create archive SHA/size differs')
    application_names=['policy-tree/'+name for name in request['_bound_observation']['binding']['records']]
    def verify(stream):return modules['reader'].verify_stream(stream,before,max_archive_bytes=request['max_archive_bytes'],max_source_bytes=request['max_source_bytes'],max_metadata_bytes=request['max_metadata_bytes'],max_evidence_bytes=request['max_evidence_bytes'],application_names=application_names,deadline=deadline)
    independent=transport.call('stream',request,deadline=deadline,consumer=verify,
        expected_index_sha256=refs['index']['sha256'],expected_archive_sha256=created['archive_sha256'],expected_archive_bytes=created['archive_bytes'])
    require(independent==primary,'independent full stream differs from primary verification')
    for key in ('node_count','file_count','regular_bytes'):require(created.get(key)==independent.get(key),'create/stream count differs: '+key)
    require(independent['origin_map']==controls['origin'],'independent origin map differs')
    modules['policy'].validate_archived_application_records(request,request['_bound_observation']['binding'],independent['origin_map'],None,deadline,
        record_reader=lambda _path,name,_deadline,_maximum:independent['application_records'][name])
    fresh=transport.call('inspect',request,deadline=deadline)
    stability=store.compare_source_inventories(before,fresh)
    recheck();guard.check();check(deadline)
    provenance=dict(schema='z1_vega_resident_preservation_provenance_v1',status='PASS',request_identity_sha256=request_identity(request),create_transport_exit=0,
        create=created,source_stability=stability,helper_package=request['helper_package_reference'],capacity_ledger=request['capacity_ledger_reference'],
        ledger=request.get('_ledger'),ledger_body_b64=request.get('_ledger_body_b64'),release_sha256=request['cpu_supervisor_release']['sha256'],
        observation_binding=request['_bound_observation']['binding'],
        request={k:v for k,v in request.items() if not k.startswith('_')},
        independent_stream=dict(status='PASS',archive_sha256=independent['archive_sha256'],archive_bytes=independent['archive_bytes'],transport_exit=0,complete_eof=True))
    prov_ref=publish(transport,request,'preservation-provenance.json',provenance,deadline)
    full=normalized_receipt(request,independent,before,fresh,created,primary_reference=refs['primary'],provenance_reference=prov_ref)
    full_ref=publish(transport,request,'consumer-receipt.json',full,deadline)
    recheck();check(deadline)
    compact=dict(schema='z1_vega_resident_compact_receipt_v1',status='PASS',backend_mode=MODE,policy=request['policy'],scope=request['scope'],bank=request.get('bank'),
        independent_backup={k:full_ref[k] for k in ('path','sha256')},archive_sha256=independent['archive_sha256'],archive_bytes=independent['archive_bytes'],
        provenance=prov_ref,create=created,create_transport_exit=0,helper_package=request['helper_package_reference'],capacity_ledger=request['capacity_ledger_reference'],
        request_identity_sha256=request_identity(request),technical_qualification=False,scientific_qualification=False,raw_tree_extracted=False)
    guard.write(request['policy_id']+'-'+request['job_id']+'-receipt.json',json_bytes(compact))
    return compact


def remote_authority(request,deadline):
    """Compile the actual physical supervisor, then use its unmodified gates."""
    check(deadline);modules=dependencies()
    release_path=plain(request['remote_supervisor_release']['path'])
    require(release_path in {Path(CPU_ROOT)/'admission-staged-integration002'/phase/'release.json' for phase in ('first','remaining','science')},'remote release outside exact phase admission root')
    release=parse_json(bound_bytes(request['remote_supervisor_release'],deadline))
    require(request['remote_supervisor_release']['sha256']==request['cpu_supervisor_release']['sha256'],'actual remote release SHA differs')
    manifest_ref=release.get('manifest');body=bound_bytes(manifest_ref,deadline)
    require(hashlib.sha256(body).hexdigest()==request['cpu_supervisor_manifest']['sha256'],'actual remote manifest SHA differs')
    manifest=parse_json(body);preservation(request,manifest)
    supervisor=Path(CPU_ROOT)/'package-staged-integration002/supervisor/cpu_supervisor.py'
    source=read_bytes(supervisor,131072,deadline)
    require(hashlib.sha256(source).hexdigest()==SUPERVISOR_SHA,'actual physical supervisor source SHA differs')
    module=types.ModuleType('resident_actual_cpu_supervisor');module.__file__=str(supervisor)
    exec(compile(source,str(supervisor),'exec'),module.__dict__)
    loaded=module.load_release(release_path);require(loaded==manifest,'actual load_release manifest differs')
    selected=[]
    for index,slot in enumerate(loaded['slots']):
        intent=slot['intent']
        if (intent['policy'],intent['scope'],None if intent['scope']=='own_cpu32' else intent['evaluation_seed'],intent['output_root'],slot['private_root'])==(request['policy'],request['scope'],request.get('bank'),request['source_root'],request['private_root']):selected.append(index)
    require(len(selected)==1,'actual exact policy/scope/bank/roots not in release')
    module.require_selected(loaded,selected[0]);check(deadline)
    authority=module.bound(loaded['authority']);phase={'first_own32':'first','remaining_own32':'remaining','science160':'science'}.get(authority.get('execution_phase'))
    require(phase is not None and release_path==Path(CPU_ROOT)/'admission-staged-integration002'/phase/'release.json','release path differs from actual selected authority phase')
    intent=loaded['slots'][selected[0]]['intent']
    require(request['remote_python']==REMOTE_PYTHON,'approved lexical CPU Python required')
    resolved=Path(request['remote_python']).resolve(strict=True)
    require(str(resolved)==RESOLVED_PYTHON and intent['python']==dict(path=RESOLVED_PYTHON,sha256=PYTHON_SHA),'selected actual interpreter binding differs')
    # The approved .venv alias is deliberately resolved, never rejected as a
    # source symlink. Its resolved binary has a separate exact identity check.
    require(hashlib.sha256(read_bytes(resolved,64*MIB,deadline)).hexdigest()==PYTHON_SHA,'resolved CPU interpreter SHA differs')
    return manifest


def quota_headroom():
    """Ceph-owned quota bytes, never substitute filesystem-wide free space."""
    root='/cephhome/eunikhilr'
    maximum=int(os.getxattr(root,'ceph.quota.max_bytes'))
    used=int(os.getxattr(root,'ceph.dir.rbytes'))
    require(maximum>0 and used>=0,'finite actual Ceph quota required')
    return maximum-used


def _allocation_usage(destination,deadline):
    """Count every owned partial/control/archive byte without following links."""
    total=0;archive=0;nodes=0
    if not os.path.lexists(destination):return dict(total=0,archive=0,provenance=0)
    def walk(path):
        nonlocal total,archive,nodes
        check(deadline)
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            with os.scandir(fd) as entries:
                for entry in entries:
                    nodes+=1;require(nodes<=8192,'operation output node cap exceeded')
                    info=entry.stat(follow_symlinks=False);child=path/entry.name
                    if stat.S_ISDIR(info.st_mode):walk(child)
                    else:
                        require(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'operation output must be new separate regular files')
                        total+=info.st_size
                        if child==destination/'terminal-policy-full.tar.gz':archive+=info.st_size
        finally:os.close(fd)
    walk(destination)
    return dict(total=total,archive=archive,provenance=total-archive)


def authenticate_credit(request,slot,role,item,compact,deadline):
    """Authenticate one root-bound credit using current full remote closure.

    Compact proof bytes are independently hash-bound on the Mac, then checked
    here against all retained Vega objects; only this action can authorize credit.
    """
    require(compact.get('schema')=='z1_vega_resident_compact_receipt_v1' and compact.get('status')=='PASS','completed Mac compact verification required')
    require(compact.get('backend_mode')==MODE,'credit backend differs')
    full=parse_json(bound_bytes(compact['independent_backup'],deadline,META))
    require(full.get('schema')=='z1_horizon_terminal_multiroot_archive_receipt_v1' and full.get('status')=='PASS' and full.get('independent_readback')=='PASS' and full.get('readback_platform')=='darwin','full independent Mac verification required')
    require(full.get('independent_stream',{}).get('complete_eof') is True and full['independent_stream'].get('transport_exit')==0,'complete independent stream exit required')
    for key in ('archive_sha256','archive_bytes'):
        require(compact.get(key)==full.get(key)==full['independent_stream'].get(key),'credit independent archive identity differs')
    require(compact['helper_package']['sha256']==request['helper_package_reference']['sha256'] and full['helper_package']==compact['helper_package'],'credit helper binding differs')
    provenance=parse_json(bound_bytes({k:compact['provenance'][k] for k in ('path','sha256')},deadline,META))
    require(full.get('preservation_provenance')==compact['provenance'] and provenance.get('status')=='PASS','credit provenance closure differs')
    require(provenance.get('request_identity_sha256')==compact.get('request_identity_sha256') and provenance.get('create_transport_exit')==compact.get('create_transport_exit')==0,'credit request/process exit proof differs')
    previous_request=provenance.get('request')
    require(isinstance(previous_request,dict) and request_identity(previous_request)==compact['request_identity_sha256'],'credit exact original request missing')
    remote_authority(previous_request,deadline)
    created=compact['create'];require(created==provenance.get('create') and created.get('status')=='COMPLETE','credit complete create response required')
    destination=plain(created['destination']);require(plain(str(dependencies()['store'].BACKUP_ROOT)) in destination.parents,'credit destination outside fixed backup root')
    refs=created['controls'];controls={key:parse_json(bound_bytes({k:ref[k] for k in ('path','sha256')},deadline,META)) for key,ref in refs.items()}
    require(set(controls)=={'attempt','before','after','primary','origin','index'},'credit full bound control set required')
    require(controls['index'].get('status')=='ARCHIVE_BOUND' and controls['index'].get('controls')=={k:v for k,v in refs.items() if k!='index'},'credit index closure differs')
    dependencies()['store'].compare_source_inventories(controls['before'],controls['after'])
    primary=controls['primary'];require(primary.get('origin_map')==controls['origin']==full.get('node_metadata'),'credit full metadata proof differs')
    for key in ('archive_sha256','archive_bytes'):
        require(created[key]==primary[key]==full[key]==controls['index'][key],'credit primary/create/index identity differs')
    require(full['job_before']==controls['before']['job'] and full['job_after']['job_id']==controls['before']['job']['job_id'],'credit scheduler proof differs')
    for key in ('policy','scope','bank','shared_source'):
        require(full.get(key)==previous_request.get(key),'credit full request binding differs: '+key)
    require(full.get('run_root')==previous_request['source_root'] and full.get('private_root')==previous_request['private_root'] and full.get('archive_path')==created['archive_path'],'credit full source/archive location differs')
    require(full.get('primary_verification')==refs['primary'] and full.get('raw_tree_extracted') is False,'credit primary/restore semantics differ')
    for prefix,map_key,tree_key in [('policy-tree','members','tree_sha256'),('private-tree','private_members','private_tree_sha256')]:
        members={name[len(prefix)+1:]:row['sha256'] for name,row in primary['origin_map'].items() if name.startswith(prefix+'/') and row['type']=='file'}
        require(full.get(map_key)==members and full.get(tree_key)==digest(members),'credit complete regular member/tree proof differs')
    dependencies()['policy'].validate_archived_application_records(previous_request,provenance['observation_binding'],primary['origin_map'],None,deadline,
        record_reader=lambda _path,name,_deadline,_maximum:primary['application_records'][name])

    if slot is not None:
        require((compact['policy']['policy_id'],compact['scope'],compact.get('bank'))==(slot['policy_id'],slot['scope'],slot['bank']),'credit ledger slot differs')
        require(full['run_root']==slot['run_root'] and full['private_root']==slot['private_root'],'credit source roots differ')
    # Existing proof records must bind the same fixed rule and closed map. The
    # previous ledger is a concrete full Vega provenance object in this closure.
    prior=provenance.get('ledger')
    require(isinstance(prior,dict),'credit previous complete ledger required')
    dependencies()['capacity'].remaining_budget(prior)
    prior_body=base64.b64decode(provenance.get('ledger_body_b64',''),validate=True)
    require(len(prior_body)<=EVIDENCE and parse_json(prior_body)==prior and hashlib.sha256(prior_body).hexdigest()==full['capacity_ledger']['sha256']==previous_request['capacity_ledger_reference']['sha256'],'credit previous ledger hash/bytes differ')
    archive_path=plain(created['archive_path']);archive_info=archive_path.lstat()
    require(stat.S_ISREG(archive_info.st_mode) and archive_info.st_nlink==1 and archive_info.st_size==created['archive_bytes'],'credited separate archive missing/changed')
    require(hashlib.sha256(read_bytes(archive_path,201326592,deadline)).hexdigest()==created['archive_sha256'],'credited separate archive SHA changed')
    require(prior['request_identity_sha256']==compact['request_identity_sha256'] and prior['release_sha256']==provenance.get('release_sha256'),'credit release/ledger proof differs')
    source_nodes=controls['before']['nodes']
    path=plain(item['path']);info=path.lstat()
    require(stat.S_ISREG(info.st_mode),'credited object missing/not regular')
    actual=dict(bytes=info.st_size,device=info.st_dev,inode=info.st_ino,link_count=info.st_nlink)
    require(all(actual[k]==item[k] for k in actual),'credited retained identity/bytes changed')
    require(hashlib.sha256(read_bytes(path,201326592,deadline)).hexdigest()==item['sha256'],'credited retained SHA changed')
    if role=='raw':
        matches=[row for row in source_nodes.values() if row.get('type')=='file' and row.get('source_path')==str(path)]
        require(len(matches)==1 and dependencies()['store']._stable(dependencies()['store']._metadata(path))==dependencies()['store']._stable({k:v for k,v in matches[0].items() if k!='sha256'}),'credited raw metadata/identity changed')
        require(len(matches)==1 and matches[0]['sha256']==item['sha256'] and matches[0]['bytes']==item['bytes'] and all(matches[0]['source_identity'][k]==item[k] for k in ('device','inode','link_count')),'credited raw object absent from full verified origin')
    elif role=='archive':
        require(str(path)==created['archive_path'] and item['sha256']==created['archive_sha256'] and item['bytes']==created['archive_bytes'] and item['link_count']==1,'credited archive differs')
    else:
        allowed=[*refs.values(),compact['provenance'],compact['independent_backup']]
        require(any(ref['path']==str(path) and ref['sha256']==item['sha256'] for ref in allowed),'provenance/shared credit lacks complete retained proof')
    return dict(status='VERIFIED',path=str(path),sha256=item['sha256'],**actual)


def remote_validate(request,deadline,credit_proofs):
    require(request.get('schema')==SCHEMA and request.get('backend_mode')==MODE,'resident remote request required')
    require(request.get('ssh_host')=='vega' and request.get('quota_root')=='/cephhome/eunikhilr','closed remote host/quota required')
    remote_authority(request,deadline)
    contract=parse_json(bound_bytes(request['remote_capacity_contract_reference'],deadline))
    # Local ledger JSON bytes are transported unchanged; no formatting-derived
    # identity and no caller-authenticated boolean can substitute for its hash.
    ledger_body=base64.b64decode(request['_ledger_body_b64'],validate=True)
    require(len(ledger_body)<=EVIDENCE and hashlib.sha256(ledger_body).hexdigest()==request['capacity_ledger_reference']['sha256'],'transported ledger SHA differs')
    ledger=parse_json(ledger_body);capacity_contract(request,contract,ledger)
    modules=dependencies();modules['contract'].validate_paths(request);modules['store']._bounds(request)
    require(request['remote_archive_destination']==str(modules['store'].validate_destination(request)),'remote destination differs')
    for slot,role,item in modules['capacity'].retained_objects(ledger):
        key=item['proof']['sha256'];raw=base64.b64decode(credit_proofs.get(key,''),validate=True)
        require(len(raw)<=MIB and hashlib.sha256(raw).hexdigest()==key,'hash-bound compact credit proof missing')
        authenticate_credit(request,slot,role,item,parse_json(raw),deadline)
    return ledger


def remote_dispatch(packet,*,deadline,sink=None):
    """Closed operations; every effect follows actual authority/ledger checks."""
    action=packet.get('action')
    require(action in ('create','readcontrol','stream','inspect','publish'),'closed remote action required')
    request=packet['request'];ledger=remote_validate(request,deadline,packet.get('credit_proofs',{}))
    modules=dependencies();store=modules['store'];capacity=modules['capacity'];destination=store.validate_destination(request)
    chosen=next(row for row in ledger['slots'] if (row['policy_id'],row['scope'],row['bank'])==(request['policy_id'],request['scope'],request.get('bank')))
    residual={role:chosen['caps'][role]-sum(x['bytes'] for x in chosen['retained'][role]) for role in capacity.CAPS}
    growth=request['operation_growth_bytes'];require(type(growth) is int and 0<growth<=residual['archive']+residual['provenance'],'operation growth exceeds selected residual caps')
    budget=capacity.operation_budget(ledger,growth)
    # There are at most nine closed control filenames including failure.json.
    # Conservative per-file caps enforce each role *before* a write, even with
    # retained provenance credits; the aggregate free-reader guard enforces G.
    metadata_cap=min(request['max_metadata_bytes'],residual['provenance']//9)
    archive_cap=min(request['max_archive_bytes'],residual['archive'])
    require(metadata_cap>0 and archive_cap>0,'selected archive/control residual allocation exhausted')
    request=dict(request,future_reserve_bytes=max(request['remote_future_reserve_bytes'],budget['future_reserve_bytes']),
                 max_metadata_bytes=metadata_cap,max_archive_bytes=archive_cap)
    protected=capacity.FLOOR+max(request['future_reserve_bytes'],capacity.FUTURE)
    def free_reader():
        usage=_allocation_usage(destination,deadline)
        require(usage['total']<=growth and usage['archive']<=residual['archive'] and usage['provenance']<=residual['provenance'],'cumulative operation/role growth cap exceeded')
        actual=quota_headroom();check(deadline)
        require(actual>=protected+growth-usage['total'],'fresh full remaining capacity preflight failed')
        return min(actual,protected+growth-usage['total'])
    free_reader()
    if action=='create':
        require(not os.path.lexists(destination),'exclusive archive destination already exists')
        result=store.create_archive(request,destination,inventory_reader=modules['remote'].inventory,free_reader=free_reader,deadline=deadline)
    elif action=='inspect':result=store.inspect_source(request,inventory_reader=modules['remote'].inventory,deadline=deadline)
    elif action=='stream':
        require(sink is not None,'binary stream sink required')
        allowed=('expected_index_sha256','expected_archive_sha256','expected_archive_bytes')
        require(set(packet.get('payload',{}))==set(allowed),'exact stream bindings required')
        return store.stream_back(request,destination,sink,deadline=deadline,**packet['payload'])
    elif action=='readcontrol':
        ref=packet['payload']['reference'];path=plain(ref['path'])
        require(path.parent==destination and path.name in ('attempt.json','before.json','after.json','verification.json','origin.json','index.json','preservation-provenance.json','consumer-receipt.json'),'closed control name/path required')
        result=dict(body_b64=base64.b64encode(bound_bytes({k:ref[k] for k in ('path','sha256')},deadline,META)).decode())
    else:
        payload=packet['payload'];name=payload['name']
        require(name in ('preservation-provenance.json','consumer-receipt.json'),'closed publication name required')
        body=base64.b64decode(payload['body_b64'],validate=True);require(len(body)<=request['max_metadata_bytes'],'publication residual/control cap exceeded')
        # A normalized consumer receipt may be published only with its bound full
        # provenance; source/current request must still be authentic at this RPC.
        value=parse_json(body)
        if name=='consumer-receipt.json':
            require(value.get('backend_mode')==MODE and value.get('policy')==request['policy'] and value.get('private_root')==request['private_root'] and value.get('run_root')==request['source_root'],'published receipt identity differs')
            bound_bytes({k:value['preservation_provenance'][k] for k in ('path','sha256')},deadline,META)
        result=dict(store.publish_control(destination/name,body,hashlib.sha256(body).hexdigest(),free_reader=free_reader,deadline=deadline,future_reserve_bytes=request['future_reserve_bytes']),path=str(destination/name))
    free_reader();check(deadline)
    return result


def remote_program(reference,seconds):
    require(type(seconds) is int and 0<seconds<=560,'bounded remote bootstrap wall required')
    return BOOTSTRAP+"\nimport signal,time\ndef bootstrap_alarm(_signum,_frame): raise TimeoutError('outer remote bootstrap deadline expired')\n"+\
        "signal.signal(signal.SIGALRM,bootstrap_alarm)\nremote_limit=time.monotonic()+"+str(seconds)+"\nsignal.setitimer(signal.ITIMER_REAL,"+str(seconds)+")\n"+\
        "m=load_verified_package("+repr(reference)+")['vega_policy_archive.resident_archive']\nsignal.setitimer(signal.ITIMER_REAL,0)\nm.remote_main(outer_deadline=remote_limit)\n"


class SSHTransport:
    """Closed Vega SSH argv and source-only pinned bootstrap; no shell payload."""
    def __init__(self,credit_proofs=None):self.credit_proofs=credit_proofs or {}
    def call(self,action,request,*,deadline,consumer=None,**payload):
        check(deadline)
        seconds=min(560,int(deadline-time.monotonic())-1)
        require(seconds>0,'deadline expired before remote dispatch')
        packet=dict(action=action,request=request,payload=payload,credit_proofs=self.credit_proofs,remote_wall_seconds=seconds)
        body=json_bytes(packet);require(len(body)<=8*MIB,'remote input packet cap exceeded')
        reference=request['remote_helper_package_reference']
        program=remote_program(reference,seconds)
        require(request['ssh_host']=='vega' and request['remote_python']==REMOTE_PYTHON,'closed approved SSH target required')
        argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','vega',
              shlex.join([REMOTE_PYTHON,'-B','-I','-S','-u','-c',program])]
        result=dependencies()['transport'].run_process(argv,body,deadline=deadline,
            stdout_cap=request['max_archive_bytes'] if action=='stream' else 8*MIB,consumer=consumer)
        if action=='stream':return result['value']
        return parse_json(result['stdout'])


def _alarm(_signum,_frame):raise TimeoutError('outer resident IO wall deadline expired')


def _watchdog(deadline):
    """Main-thread POSIX outer guard interrupts blocking stat/read/write/fsync."""
    import threading
    require(threading.current_thread() is threading.main_thread(),'outer IO guard requires main thread')
    previous=signal.getsignal(signal.SIGALRM);timer=signal.getitimer(signal.ITIMER_REAL)
    require(timer==(0.0,0.0),'existing process alarm must not be replaced')
    signal.signal(signal.SIGALRM,_alarm);signal.setitimer(signal.ITIMER_REAL,max(.000001,deadline-time.monotonic()))
    return previous


def _unwatch(previous):
    signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def remote_main(*,outer_deadline=None):
    previous=None
    try:
        # The bootstrap itself has no payload read; cap/outer timeout starts
        # before reading stdin, including a peer that never closes its input.
        outer_deadline=min(time.monotonic()+560,outer_deadline) if outer_deadline is not None else time.monotonic()+560
        previous=_watchdog(outer_deadline)
        body=sys.stdin.buffer.read(8*MIB+1);require(len(body)<=8*MIB,'remote stdin byte cap exceeded')
        packet=parse_json(body);deadline=min(outer_deadline,remote_deadline(packet['remote_wall_seconds']))
        signal.setitimer(signal.ITIMER_REAL,max(.000001,deadline-time.monotonic()))
        result=remote_dispatch(packet,deadline=deadline,sink=sys.stdout.buffer)
        if packet['action']!='stream':
            data=json_bytes(result);require(len(data)<=8*MIB,'remote stdout JSON cap exceeded');sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
    except BaseException as exc:
        sys.stderr.write((type(exc).__name__+': '+str(exc))[:4096]+'\n');raise SystemExit(1)
    finally:
        if previous is not None:_unwatch(previous)


def run(request_path,output):
    """Public Mac coordinator. No archive or raw extraction is written locally."""
    started=time.monotonic();deadline=started+600;previous=_watchdog(deadline);guard=None
    try:
        request_path=plain(str(request_path));body=read_bytes(request_path,EVIDENCE,deadline);public=parse_json(body)
        require(public.get('schema')==SCHEMA,'resident request required')
        package_ref=dict(public['helper_package_reference'])
        if not Path(package_ref['path']).is_absolute():package_ref['path']=str(request_path.parent/package_ref['path'])
        loaded=verify_package(package_ref)
        # Re-enter the verified source module. An imported unverified coordinator
        # cannot cause a dependency to load before this complete package check.
        implementation=loaded['vega_policy_archive.resident_archive']
        require(plain(implementation.__file__).parent==plain(package_ref['path']).parent,'entry source/package directory differs')
        return implementation._run_verified(request_path,output,body,started,deadline)
    finally:_unwatch(previous)


def _run_verified(request_path,output,body,started,deadline):
    public=parse_json(body)
    seconds=public.get('wall_seconds');require(type(seconds) is int and 0<seconds<=600,'bounded positive wall_seconds required')
    deadline=min(deadline,started+seconds);signal.setitimer(signal.ITIMER_REAL,max(.000001,deadline-time.monotonic()))
    request=validate(public,request_path,deadline)
    require(sys.platform=='darwin','actual independent Mac readback required')
    require(plain(str(output))==plain(request['mac_control_root']),'explicit admitted Mac control root required')
    modules=dependencies();guard=modules['capacity'].ControlRoot(output,deadline=deadline);guard.check()
    # Snapshot every original binding and request before any remote effect.
    references=[dict(path=str(request_path),sha256=hashlib.sha256(body).hexdigest())]
    references += [dict(path=x['path'],sha256=x['sha256']) for x in request['_bound_extra']]
    references += [{k:request['_bound_evidence'][k] for k in ('path','sha256')}]
    references += [public[k] for k in ('helper_package_reference','capacity_contract_reference','capacity_ledger_reference')]
    def recheck():
        check(deadline)
        for ref in references:bound_bytes(ref,deadline,base=request_path.parent)
        # Includes the entry source and all helper bytes, not just the manifest.
        verify_package(dict(path=str(Path(request['helper_package_reference']['path']) if Path(request['helper_package_reference']['path']).is_absolute() else request_path.parent/request['helper_package_reference']['path']),sha256=request['helper_package_reference']['sha256']))
    recheck()
    request['_ledger_body_b64']=base64.b64encode(bound_bytes(public['capacity_ledger_reference'],deadline,base=request_path.parent)).decode()
    proofs={}
    for _,_,item in modules['capacity'].retained_objects(request['_ledger']):
        raw=bound_bytes(item['proof'],deadline,MIB,base=request_path.parent)
        proofs[item['proof']['sha256']]=base64.b64encode(raw).decode()
    transport=SSHTransport(proofs)
    try:return coordinate(request,transport,guard,deadline=deadline,recheck=recheck)
    except BaseException as exc:
        try:guard.write(request['policy_id']+'-'+request['job_id']+'-error.json',json_bytes(dict(schema='z1_vega_resident_failure_v1',status='FAILED_NO_RETRY',error_type=type(exc).__name__,error=str(exc)[:4096],request_identity_sha256=request_identity(request))))
        except BaseException:pass
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.request,args.output);sys.stdout.write(json_bytes(result).decode()+'\n')

if __name__=='__main__':main()
