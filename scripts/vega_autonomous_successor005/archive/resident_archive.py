#!/usr/bin/env python3
"""Vega-resident preservation; Linux verifier independently checks a local writer child without extraction.

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

MODE='VEGA_RESIDENT_FULL_ARCHIVE_VEGA_PROCESS_V1'
SCHEMA='z1_horizon_vega_resident_archive_request_v1'
CPU_ROOT='/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002'
REMOTE_PYTHON='/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python'
RESOLVED_PYTHON='/cephhome/eunikhilr/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'
PYTHON_SHA='d778d0b551287fb5f06c4464f78b616c086f8d42e94d8cfc209b12324cf937a1'
SUPERVISOR_SHA='565bb43027d8c9a98505603985d7a95b8f54c093c51e6cbcb2ba947f1507c4fd'
CONTROL_ROOT=CPU_ROOT+'/admission-staged-integration002/autonomous005'
EVIDENCE_ROOT=CPU_ROOT+'/evidence-autonomous005'
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
    require(not str(path).startswith('/Users/'),'Mac operative reference forbidden')
    if not path.is_absolute():
        require(base is not None and '..' not in path.parts,'bound relative path requires base');path=base/path
    body=read_bytes(path,maximum,deadline)
    require(hashlib.sha256(body).hexdigest()==reference['sha256'],'bound input SHA differs: '+str(path))
    return body


# Canonical security bootstrap. The isolated verifier region is mechanically
# generated from this literal by tests/generate_bootstrap.py. This exact source
# also runs as the local writer -c argument. Its manifest hash is the
# caller's preservation-contract hash. No stage source executes before *all*
# source bytes and compiled code objects have passed the complete inventory.
BOOTSTRAP=r'''
import hashlib,json,os,stat,sys,types
from pathlib import Path

def read_remote_packet(stream):
    """Read the admitted packet once, before any candidate module executes."""
    body=stream.read(8388609)
    if len(body)>8388608:raise ValueError('remote stdin byte cap exceeded')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    def invalid(_value):raise ValueError('nonfinite JSON')
    packet=json.loads(body,object_pairs_hook=pairs,parse_constant=invalid)
    if not isinstance(packet,dict) or not isinstance(packet.get('request'),dict):raise ValueError('remote packet/request object required')
    seconds=packet.get('remote_wall_seconds')
    if type(seconds)is not int or not 0<seconds<=560:raise ValueError('bounded remote wall seconds required')
    return packet


def authenticate_package_authority(reference,request,base,remote=False):
    """Trusted stdlib preflight; candidate bytes cannot supply this decision."""
    def need(ok,message):
        if not ok:raise ValueError('package preservation authority: '+message)
    def bound(spec):
        need(isinstance(spec,dict) and set(spec)=={'path','sha256'},'exact bound reference required')
        path=Path(spec['path'])
        if not path.is_absolute():path=Path(base)/path
        need(path.is_absolute() and '..' not in path.parts and not str(path).startswith('/Users/'),'canonical non-Mac bound path required')
        for parent in reversed(path.parents):need(stat.S_ISDIR(parent.lstat().st_mode),'bound parent symlink forbidden')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as stream:
            before=os.fstat(stream.fileno());need(stat.S_ISREG(before.st_mode) and before.st_size<=8388608,'bounded regular authority required')
            body=stream.read(8388609);after=os.fstat(stream.fileno())
        fields=('st_mode','st_dev','st_ino','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
        need(len(body)<=8388608 and tuple(getattr(before,k) for k in fields)==tuple(getattr(after,k) for k in fields),'authority descriptor drift')
        need(hashlib.sha256(body).hexdigest()==spec['sha256'],'authority SHA differs')
        def pairs(items):
            result={}
            for key,value in items:need(key not in result,'duplicate authority JSON key');result[key]=value
            return result
        def invalid(_value):raise ValueError('nonfinite authority JSON')
        return json.loads(body,object_pairs_hook=pairs,parse_constant=invalid)
    try:
        need(request.get('schema')=='z1_horizon_vega_resident_archive_request_v1','resident request required')
        if request.get('kind')=='CONTROL_ONLY':
            control='/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002/admission-staged-integration002/autonomous005'
            need(reference==request['helper_package_reference']==request['remote_helper_package_reference'],'control helper reference differs')
            ref=request['preservation_authority']
            need(str(Path(ref['path']).parent)==control and Path(ref['path']).name=='preservation-authority.json','closed root preservation authority required')
            authority=bound(ref)
            need(authority==dict(schema='z1_vega_control_preservation_authority_v1',backend_mode='VEGA_RESIDENT_FULL_ARCHIVE_VEGA_PROCESS_V1',helper_package=reference,backup_root='/cephhome/eunikhilr/campaign-backups/z1-horizon-evaluation-20261027-autonomous005',control_root=control),'root source preservation authority differs')
            return
        need(request['backend_mode']=='VEGA_RESIDENT_FULL_ARCHIVE_VEGA_PROCESS_V1','backend mode differs')
        local=request['helper_package_reference'];remote_ref=request['remote_helper_package_reference']
        expected=remote_ref if remote else dict(local,path=str(Path(local['path']) if Path(local['path']).is_absolute() else Path(base)/local['path']))
        need(reference==expected and local['sha256']==remote_ref['sha256'],'candidate differs from requested package')
        release_ref=request['remote_supervisor_release'] if remote else request['cpu_supervisor_release']
        release=bound(release_ref)
        need(release.get('status')=='ADMITTED_ONCE' and release.get('manifest',{}).get('sha256')==request['cpu_supervisor_manifest']['sha256'],'actual release/manifest closure differs')
        need(request['remote_supervisor_release']['sha256']==request['cpu_supervisor_release']['sha256'],'local/remote release differs')
        manifest=bound(release['manifest'] if remote else request['cpu_supervisor_manifest'])
        need(manifest.get('schema')=='z1_horizon_cpu_supervisor_v1','actual CPU manifest required')
        need(manifest.get('supervisor_sha256')=='565bb43027d8c9a98505603985d7a95b8f54c093c51e6cbcb2ba947f1507c4fd','actual CPU supervisor binding differs')
        backup='/cephhome/eunikhilr/campaign-backups/z1-horizon-evaluation-20261027-autonomous005'
        capacity=request['remote_capacity_contract_reference']
        need(request['capacity_contract_reference']['sha256']==capacity['sha256'],'local/remote capacity differs')
        expected_contract=dict(schema='z1_vega_resident_preservation_contract_v1',mode='VEGA_RESIDENT_FULL_ARCHIVE_VEGA_PROCESS_V1',backend_package=remote_ref,capacity_contract=capacity,backup_root=backup,control_root=request['control_root'])
        need(request['remote_archive_root']==backup and manifest.get('preservation_contract')==expected_contract,'actual manifest preservation contract differs')
        cpu='/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002'
        need(request['remote_supervisor_release']['path'] in {cpu+'/admission-staged-integration002/'+phase+'/release.json' for phase in ('remaining','science')},'closed actual phase release required')
    except (KeyError,TypeError) as exc:raise ValueError('package preservation authority incomplete') from exc


def load_verified_package(reference,request,base,remote=False):
    authenticate_package_authority(reference,request,base,remote)
    def need(ok,msg):
        if not ok:raise ValueError(msg)
    def read(path,limit):
        need(path.is_absolute() and '..' not in path.parts and not str(path).startswith('/Users/'),'absolute non-Mac package path required')
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
    need(set(manifest)=={'schema','entrypoint','files'} and manifest['schema']=='z1_vega_policy_archive_package_v1' and manifest['entrypoint']=='independent_verify.py','package manifest schema differs')
    expected={'archive/__init__.py','archive/compat/__init__.py','archive/compat/archive_contract.py','archive/compat/policy_archive.py','archive/compat/remote_policy_archive.py','archive/stream_reader.py','archive/resident_store.py','archive/capacity.py','archive/transport.py','archive/resident_archive.py','reservation.py','control_preservation.py','independent_verify.py'}
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
    names=[('archive/__init__.py','vega_policy_archive'),('archive/compat/__init__.py','vega_policy_archive.compat'),('archive/stream_reader.py','vega_policy_archive.stream_reader'),('archive/resident_store.py','vega_policy_archive.resident_store'),('archive/capacity.py','vega_policy_archive.capacity'),('archive/transport.py','vega_policy_archive.transport'),('reservation.py','vega_policy_archive.reservation'),('control_preservation.py','vega_policy_archive.control_preservation'),('archive/resident_archive.py','vega_policy_archive.resident_archive')]
    result={}
    for relative,name in names:
        module=types.ModuleType(name);module.__file__=str(root/relative)
        module.__package__=name if relative.endswith('__init__.py') else name.rpartition('.')[0]
        if relative.endswith('__init__.py'):module.__path__=[str((root/relative).parent)]
        sys.modules[name]=module
        exec(codes[relative],module.__dict__);result[name]=module
    return result
'''


def verify_package(reference,*,authority_request,base):
    namespace={};exec(compile(BOOTSTRAP,'<bound-package-bootstrap>','exec'),namespace)
    return namespace['load_verified_package'](reference,authority_request,str(base))


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
        backup_root=str(store.BACKUP_ROOT),control_root=request['control_root'])
    require(manifest.get('preservation_contract')==expected,'actual manifest preservation contract differs')
    require(request['backend_mode']==MODE and request['remote_archive_root']==str(store.BACKUP_ROOT),'closed backend mode/root required')
    for local,remote in [('helper_package_reference','remote_helper_package_reference'),('capacity_contract_reference','remote_capacity_contract_reference')]:
        require(request[local]['sha256']==request[remote]['sha256'],'local/remote helper/capacity SHA differs')
    return expected


def capacity_contract(request,contract,ledger):
    cap=dependencies()['capacity']
    require(contract==dict(schema='z1_vega_resident_capacity_contract_v1',rule_version=cap.RULE,
        slots=cap.layout_rows(),slot_caps=cap.CAPS,shared_caps=cap.SHARED,
        immutable_reserve_bytes=cap.IMMUTABLE,control_root=CONTROL_ROOT,
        shared_controls_bytes=cap.SHARED['controls'],working_metadata_bytes=88*MIB),
        'capacity contract differs from fixed allocations')
    cap.remaining_budget(ledger)
    reservation=parse_json(bound_bytes(request['reservation_reference'],time.monotonic()+30))
    require(reservation['ledger_sha256']==digest(ledger) and reservation['ledger']==ledger,'frozen preissue ledger binding differs')
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
    control_root=plain(request['control_root']);require(control_root==Path(CONTROL_ROOT),'exact Vega control root required')
    for root in (plain(request['source_root']),plain(request['private_root'])):
        require(control_root!=root and control_root not in root.parents and root not in control_root.parents,'Vega controls must be separate from source roots')
    package=bound_bytes(request['helper_package_reference'],deadline,base=request_path.parent)
    require(len(package)<=131072,'package manifest cap exceeded')
    contract=parse_json(bound_bytes(request['capacity_contract_reference'],deadline,base=request_path.parent))
    ledger=parse_json(bound_bytes(request['capacity_ledger_reference'],deadline,base=request_path.parent))
    capacity_contract(original,contract,ledger)
    require(request['remote_supervisor_release']['sha256']==request['cpu_supervisor_release']['sha256'],'remote release differs from local actual release')
    release=plain(request['remote_supervisor_release']['path'])
    require(release in {Path(CPU_ROOT)/'admission-staged-integration002'/phase/'release.json' for phase in ('remaining','science')},'actual closed phase admission release required')
    destination=modules['store'].validate_destination(request)
    require(request['remote_archive_destination']==str(destination),'derived remote destination differs')
    # Prevent post-validation identity changes, including previously omitted fields.
    request=original
    request.update({k:v for k,v in legacy.items() if k.startswith('_')})
    request['_request_identity_sha256']=request_identity(original)
    request['_ledger']=ledger;request['_capacity_contract']=contract
    return request


def normalized_receipt(request,verified,before,after,created,*,primary_reference,provenance_reference,protocol_stages,readback_identity):
    origins=verified['origin_map']
    def members(prefix):return {name[len(prefix)+1:]:row['sha256'] for name,row in origins.items() if name.startswith(prefix+'/') and row['type']=='file'}
    raw=members('policy-tree');private=members('private-tree')
    return dict(schema='z1_horizon_terminal_multiroot_archive_receipt_v1',status='PASS',independent_readback='PASS',
        archive_only=True,readback_platform=readback_identity['platform'],readback_hostname=readback_identity['hostname'],
        technical_qualification=False,scientific_qualification=False,policy=request['policy'],scope=request['scope'],bank=request.get('bank'),
        run_root=request['source_root'],private_root=request['private_root'],members=raw,tree_sha256=digest(raw),
        private_members=private,private_tree_sha256=digest(private),training_manifest_sha256=request.get('training_manifest_sha256'),
        archive_path=created['archive_path'],archive_sha256=verified['archive_sha256'],archive_bytes=verified['archive_bytes'],
        job_before=before['job'],job_after=after['job'],shared_source=request['shared_source'],node_metadata=origins,raw_tree_extracted=False,
        backend_mode=MODE,remote_archive_location=created['destination'],primary_verification=primary_reference,
        independent_stream=dict(status='PASS',archive_sha256=verified['archive_sha256'],archive_bytes=verified['archive_bytes'],transport_exit=0,complete_eof=True),
        helper_package=request['helper_package_reference'],capacity_ledger=request['capacity_ledger_reference'],
        preservation_provenance=provenance_reference,protocol_stages=protocol_stages)


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
    """Serialize the complete operation on the admitted Vega control root."""
    with guard.lease():
        return _coordinate(request,transport,guard,deadline=deadline,recheck=recheck)


def _coordinate(request,transport,guard,*,deadline,recheck):
    """The only successful path; callback transport must guarantee exit zero."""
    modules=dependencies();store=modules['store'];destination=store.validate_destination(request)
    guard.check();check(deadline)
    created=transport.call('create',request,deadline=deadline)
    require(created.get('status')=='COMPLETE','writer archive did not complete');validate_process_pair(created['writer_identity'],process_identity(request))
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
    stream_value=stream_stage_record(request,created)
    stream_ref=control_reference(destination/'stream-stage.json',stream_value)
    require(control_read(transport,request,stream_ref,deadline)==stream_value,'completed stream stage differs')
    inspected=transport.call('inspect',request,deadline=deadline,completed_stream_reference=stream_ref)
    fresh=inspected['inventory'];stability=store.compare_source_inventories(before,fresh)
    fresh_value=fresh_stage_record(request,stream_ref,fresh,stability)
    fresh_ref=control_reference(destination/'fresh-source-stage.json',fresh_value)
    require(inspected.get('stage')==fresh_ref and control_read(transport,request,fresh_ref,deadline)==fresh_value,'fresh source stage differs')
    stages=dict(stream=stream_ref,fresh=fresh_ref)
    recheck();guard.check();check(deadline)
    readback=process_identity(request)
    provenance=provenance_record(request,created,independent,stability,stages,readback)
    prov_ref=publish(transport,request,'preservation-provenance.json',provenance,deadline)
    full=normalized_receipt(request,independent,before,fresh,created,primary_reference=refs['primary'],provenance_reference=prov_ref,protocol_stages=stages,readback_identity=readback)
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
    require(release_path in {Path(CPU_ROOT)/'admission-staged-integration002'/phase/'release.json' for phase in ('remaining','science')},'remote release outside exact phase admission root')
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
    reservation=parse_json(bound_bytes(request['reservation_reference'],deadline));require(reservation['index']==selected[0],'reservation index differs from actual CPU manifest slot')
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


def _credit_closure(request,compact,deadline):
    """Authenticate a compact proof once; retain its closure only for its group.

    Compact proof bytes are independently hash-bound by the verifier, then checked
    here against all retained Vega objects; only this action can authorize credit.
    """
    require(compact.get('schema')=='z1_vega_resident_compact_receipt_v1' and compact.get('status')=='PASS','completed Linux compact verification required')
    require(compact.get('backend_mode')==MODE,'credit backend differs')
    full=parse_json(bound_bytes(compact['independent_backup'],deadline,META))
    require(full.get('schema')=='z1_horizon_terminal_multiroot_archive_receipt_v1' and full.get('status')=='PASS' and full.get('independent_readback')=='PASS' and full.get('readback_platform')=='linux','full independent Linux verification required')
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

    # Existing proof records must bind the same fixed rule and closed map. The
    # previous ledger is a concrete full Vega provenance object in this closure.
    prior=provenance.get('ledger')
    require(isinstance(prior,dict),'credit previous complete ledger required')
    dependencies()['capacity'].remaining_budget(prior)
    previous_request=dict(previous_request,_ledger=prior,_ledger_body_b64=provenance.get('ledger_body_b64'),_bound_observation=dict(binding=provenance['observation_binding']))
    protocol_created,protocol_controls,protocol_fresh,protocol_stability,protocol_stages=_publication_inputs(previous_request,previous_request,destination,deadline)
    readback=_validate_provenance(provenance,previous_request,protocol_created,protocol_controls['primary'],protocol_stability,protocol_stages)
    expected_full=normalized_receipt(previous_request,protocol_controls['primary'],protocol_controls['before'],protocol_fresh,protocol_created,
        primary_reference=protocol_created['controls']['primary'],provenance_reference=compact['provenance'],protocol_stages=protocol_stages,readback_identity=readback)
    require(full==expected_full,'credit exact normalized protocol closure differs')
    prior_body=base64.b64decode(provenance.get('ledger_body_b64',''),validate=True)
    require(len(prior_body)<=EVIDENCE and parse_json(prior_body)==prior and hashlib.sha256(prior_body).hexdigest()==full['capacity_ledger']['sha256']==previous_request['capacity_ledger_reference']['sha256'],'credit previous ledger hash/bytes differ')
    archive_path=plain(created['archive_path']);archive_info=archive_path.lstat()
    require(stat.S_ISREG(archive_info.st_mode) and archive_info.st_nlink==1 and archive_info.st_size==created['archive_bytes'],'credited separate archive missing/changed')
    require(hashlib.sha256(read_bytes(archive_path,201326592,deadline)).hexdigest()==created['archive_sha256'],'credited separate archive SHA changed')
    require(prior['preissue_intent_sha256']==provenance.get('preissue_intent_sha256') and prior['release_sha256']==provenance.get('release_sha256'),'credit preissue/release ledger proof differs')
    archive_identity=dependencies()['store']._descriptor_identity(archive_info)
    require(archive_identity==dependencies()['store']._descriptor_identity(archive_path.lstat()),'credited archive identity changed during proof authentication')
    return dict(compact=compact,full=full,created=created,refs=refs,source_nodes=controls['before']['nodes'],
                protocol_stages=protocol_stages,archive_identity=archive_identity)


def _authenticate_credit_item(slot,role,item,verified,deadline):
    check(deadline)
    compact=verified['compact'];full=verified['full'];created=verified['created'];refs=verified['refs']
    source_nodes=verified['source_nodes'];protocol_stages=verified['protocol_stages']
    if slot is not None:
        require((compact['policy']['policy_id'],compact['scope'],compact.get('bank'))==(slot['policy_id'],slot['scope'],slot['bank']),'credit ledger slot differs')
        require(full['run_root']==slot['run_root'] and full['private_root']==slot['private_root'],'credit source roots differ')
    path=plain(item['path']);info=path.lstat()
    require(stat.S_ISREG(info.st_mode),'credited object missing/not regular')
    actual=dict(bytes=info.st_size,device=info.st_dev,inode=info.st_ino,link_count=info.st_nlink)
    require(all(actual[k]==item[k] for k in actual),'credited retained identity/bytes changed')
    if str(path)==created['archive_path']:
        require(dependencies()['store']._descriptor_identity(info)==verified['archive_identity'],'credited archive identity changed after proof authentication')
        require(item['sha256']==created['archive_sha256'],'credited archive SHA binding differs')
    else:
        require(hashlib.sha256(read_bytes(path,201326592,deadline)).hexdigest()==item['sha256'],'credited retained SHA changed')
    if role=='raw':
        matches=[row for row in source_nodes.values() if row.get('type')=='file' and row.get('source_path')==str(path)]
        require(len(matches)==1 and dependencies()['store']._stable(dependencies()['store']._metadata(path))==dependencies()['store']._stable({k:v for k,v in matches[0].items() if k!='sha256'}),'credited raw metadata/identity changed')
        require(len(matches)==1 and matches[0]['sha256']==item['sha256'] and matches[0]['bytes']==item['bytes'] and all(matches[0]['source_identity'][k]==item[k] for k in ('device','inode','link_count')),'credited raw object absent from full verified origin')
    elif role=='archive':
        require(str(path)==created['archive_path'] and item['sha256']==created['archive_sha256'] and item['bytes']==created['archive_bytes'] and item['link_count']==1,'credited archive differs')
    else:
        allowed=[*refs.values(),compact['provenance'],compact['independent_backup'],*protocol_stages.values()]
        require(any(ref['path']==str(path) and ref['sha256']==item['sha256'] for ref in allowed),'provenance/shared credit lacks complete retained proof')
    return dict(status='VERIFIED',path=str(path),sha256=item['sha256'],**actual)



def authenticate_credit(request,slot,role,item,compact,deadline):
    """Standalone authentication; remote_validate groups shared proofs per RPC."""
    return _authenticate_credit_item(slot,role,item,_credit_closure(request,compact,deadline),deadline)


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
    groups={}
    for slot,role,item in modules['capacity'].retained_objects(ledger):
        groups.setdefault(item['proof']['sha256'],[]).append((slot,role,item))
    # Grouping is a bounded per-RPC cache: one full immutable proof/archive is
    # authenticated once, all current items are checked, then its maps released.
    # Do not retain eighteen complete inventories just to avoid repeated work.
    for key,items in groups.items():
        raw=base64.b64decode(credit_proofs.get(key,''),validate=True)
        require(len(raw)<=MIB and hashlib.sha256(raw).hexdigest()==key,'hash-bound compact credit proof missing')
        verified=_credit_closure(request,parse_json(raw),deadline)
        for slot,role,item in items:_authenticate_credit_item(slot,role,item,verified,deadline)
        del verified
    return ledger


def control_reference(path,value):
    body=json_bytes(value)
    return dict(path=str(path),sha256=hashlib.sha256(body).hexdigest(),bytes=len(body))


def stream_stage_record(request,created):
    """Server transmission evidence; this is not a verifier verdict."""
    return dict(schema='z1_vega_resident_stream_stage_v1',status='STREAM_TRANSMITTED',
        request_identity_sha256=request_identity(request),policy=request['policy'],scope=request['scope'],bank=request.get('bank'),
        job_id=request['job_id'],source_root=request['source_root'],private_root=request['private_root'],
        helper_package=request['remote_helper_package_reference'],capacity_ledger=request['capacity_ledger_reference'],
        release=request['remote_supervisor_release'],create=created)


def fresh_stage_record(request,stream_ref,inventory,stability):
    return dict(schema='z1_vega_resident_fresh_source_stage_v1',status='SOURCE_STABLE',
        request_identity_sha256=request_identity(request),completed_stream_reference=stream_ref,
        inventory=inventory,source_stability=stability)


def provenance_record(request,created,independent,stability,stages,readback):
    """Exact publication contract; independent verdict is a trusted same-account verifier assertion."""
    return dict(schema='z1_vega_resident_preservation_provenance_v1',status='PASS',request_identity_sha256=request_identity(request),create_transport_exit=0,
        create=created,source_stability=stability,helper_package=request['helper_package_reference'],capacity_ledger=request['capacity_ledger_reference'],
        preissue_descriptor=request.get('_ledger',{}).get('preissue_descriptor'),preissue_intent_sha256=request.get('_ledger',{}).get('preissue_intent_sha256'),ledger=request.get('_ledger'),ledger_body_b64=request.get('_ledger_body_b64'),release_sha256=request['cpu_supervisor_release']['sha256'],
        observation_binding=request['_bound_observation']['binding'],request={k:v for k,v in request.items() if not k.startswith('_')},
        protocol_stages=stages,independent_stream=dict(status='PASS',archive_sha256=independent['archive_sha256'],archive_bytes=independent['archive_bytes'],
            transport_exit=0,complete_eof=True,readback_platform=readback['platform'],readback_hostname=readback['hostname'],verifier_identity=readback,writer_identity=created['writer_identity']))


def _archive_closure(request,destination,index_ref,deadline):
    """Reconstruct the exact successful backend result from bound retained files."""
    require(not os.path.lexists(destination/'failure.json'),'archive has retained failure evidence')
    require(isinstance(index_ref,dict) and index_ref.get('path')==str(destination/'index.json'),'exact archive index reference required')
    index_body=bound_bytes({k:index_ref[k] for k in ('path','sha256')},deadline,META)
    require(index_ref.get('bytes')==len(index_body),'index byte binding differs')
    index=parse_json(index_body)
    require(index.get('status')=='ARCHIVE_BOUND' and index.get('archive_name')=='terminal-policy-full.tar.gz' and index.get('policy_id')==request['policy_id'] and index.get('job_id')==request['job_id'],'archive index identity differs')
    names=dict(attempt='attempt.json',before='before.json',after='after.json',primary='verification.json',origin='origin.json')
    refs=index.get('controls',{});require(set(refs)==set(names),'complete backend control closure required')
    controls={}
    for key,name in names.items():
        ref=refs[key];require(ref.get('path')==str(destination/name),'foreign backend control reference')
        body=bound_bytes({k:ref[k] for k in ('path','sha256')},deadline,META)
        require(ref.get('bytes')==len(body),'backend control bytes differ')
        controls[key]=parse_json(body)
    primary=controls['primary'];before=controls['before'];after=controls['after']
    require(primary.get('origin_map')==controls['origin'],'primary/origin closure differs')
    for inv in (before,after):
        for key in ('policy_id','scope','source_root','private_root'):require(inv.get(key)==request[key],'backend inventory request differs')
        require(inv.get('job',{}).get('job_id')==request['job_id'],'backend scheduler differs')
        dependencies()['remote'].validate_guard(request,inv['job'])
    dependencies()['store'].compare_source_inventories(before,after)
    for key in ('archive_sha256','archive_bytes'):require(index.get(key)==primary.get(key),'backend archive identity differs')
    dependencies()['policy'].validate_archived_application_records(request,request['_bound_observation']['binding'],primary['origin_map'],None,deadline,
        record_reader=lambda _path,name,_deadline,_maximum:primary['application_records'][name])
    created=dict(status='COMPLETE',destination=str(destination),archive_path=str(destination/'terminal-policy-full.tar.gz'),
        archive_sha256=primary['archive_sha256'],archive_bytes=primary['archive_bytes'],node_count=primary['node_count'],
        file_count=primary['file_count'],regular_bytes=primary['regular_bytes'],controls=dict(refs,index=index_ref),created_parents=index['created_parents'])
    created['writer_identity']=parse_json(read_bytes(destination/'writer-process.json',META,deadline))
    return created,controls


def _load_stage(destination,name,deadline):
    try:body=read_bytes(destination/name,META,deadline)
    except FileNotFoundError as exc:raise ValueError('completed '+name+' protocol stage required') from exc
    value=parse_json(body)
    require(body==json_bytes(value),'protocol stage canonical bytes differ')
    return value,control_reference(destination/name,value)


def _stream_stage(request,destination,deadline):
    value,reference=_load_stage(destination,'stream-stage.json',deadline)
    try:index_ref=value['create']['controls']['index']
    except (KeyError,TypeError) as exc:raise ValueError('completed stream stage closure required') from exc
    created,controls=_archive_closure(request,destination,index_ref,deadline)
    require(value==stream_stage_record(request,created),'completed stream stage binding differs')
    return reference,created,controls


def _publication_inputs(request,effective,destination,deadline):
    stream_ref,created,controls=_stream_stage(request,destination,deadline)
    fresh,fresh_ref=_load_stage(destination,'fresh-source-stage.json',deadline)
    inventory=fresh.get('inventory');require(isinstance(inventory,dict),'complete fresh source inventory required')
    stability=dependencies()['store'].compare_source_inventories(controls['before'],inventory)
    require(fresh==fresh_stage_record(request,stream_ref,inventory,stability),'fresh source stage binding differs')
    current=dependencies()['store'].inspect_source(effective,inventory_reader=dependencies()['remote'].inventory,deadline=deadline)
    dependencies()['store'].compare_source_inventories(inventory,current)
    created['writer_identity']=parse_json(read_bytes(destination/'writer-process.json',META,deadline))
    return created,controls,inventory,stability,dict(stream=stream_ref,fresh=fresh_ref)


def _validate_provenance(value,request,created,primary,stability,stages):
    require(isinstance(value,dict),'complete preservation provenance required')
    attestation=value.get('independent_stream',{})
    require(isinstance(attestation,dict) and attestation.get('readback_platform')=='linux' and isinstance(attestation.get('readback_hostname'),str) and 0<len(attestation['readback_hostname'])<=255,'admitted Linux readback attestation required')
    readback=attestation.get('verifier_identity',{});validate_process_pair(created['writer_identity'],readback)
    require(value==provenance_record(request,created,primary,stability,stages,readback),'exact preservation provenance closure differs')
    return readback


def remote_dispatch(packet,*,deadline,sink=None):
    """Closed operations; every effect follows actual authority/ledger checks."""
    action=packet.get('action')
    require(action in ('create','readcontrol','stream','inspect','publish'),'closed remote action required')
    request=packet['request']
    if request.get('kind')=='CONTROL_ONLY':
        from . import control_preservation
        return control_preservation.dispatch(packet,deadline=deadline,sink=sink)
    ledger=remote_validate(request,deadline,packet.get('credit_proofs',{}));original_request=request
    modules=dependencies();store=modules['store'];capacity=modules['capacity'];destination=store.validate_destination(request)
    from . import reservation as reservations
    reservation=parse_json(bound_bytes(request['reservation_reference'],deadline))
    reservations.validate_reservation(reservation,ledger,credit_request=request,deadline=deadline)
    chosen=next(row for row in ledger['slots'] if all(row[key]==reservation['slot'][key] for key in reservation['slot']))
    require((chosen['policy_id'],chosen['scope'],chosen['bank'])==(request['policy_id'],request['scope'],request.get('bank')),'reservation current slot differs')
    require(reservation['release']==request['cpu_supervisor_release'],'reservation release differs')
    request=dict(request,future_reserve_bytes=max(capacity.FUTURE,max(capacity.IMMUTABLE,reservation['R']-reservation['G'])-capacity.FLOOR),max_metadata_bytes=min(request['max_metadata_bytes'],capacity.CAPS['provenance']//13))
    def free_reader(*,pending_role=None,pending_bytes=0):
        usage=reservations.current_usage(reservation,deadline=deadline)
        actual=quota_headroom();check(deadline)
        pending={role:0 for role in reservations.CAPS}
        if pending_role is not None:
            require(pending_role in pending,'closed pending write role required');pending[pending_role]=pending_bytes
        budget=reservations.check_reservation(reservation,usage,free_bytes=actual,pending=pending)
        # The same observed usage checks the pending individual role and G before
        # the store may create a file or write its next bounded piece.
        return min(actual,budget['protected_bytes']+reservation['G']-budget['consumed_bytes'])
    free_reader.check_write=lambda role,size:free_reader(pending_role=role,pending_bytes=size)
    free_reader()
    if action=='create':
        require(not os.path.lexists(destination),'exclusive archive destination already exists')
        result=store.create_archive(request,destination,inventory_reader=modules['remote'].inventory,free_reader=free_reader,deadline=deadline)
        result['writer_identity']=process_identity(request,role='writer')
        writer=control_reference(destination/'writer-process.json',result['writer_identity'])
        store.publish_control(destination/'writer-process.json',json_bytes(result['writer_identity']),writer['sha256'],free_reader=free_reader,deadline=deadline,future_reserve_bytes=request['future_reserve_bytes'])
    elif action=='inspect':
        payload=packet.get('payload',{});require(set(payload)=={'completed_stream_reference'},'completed stream protocol reference required before fresh inspection')
        stream_ref,created,controls=_stream_stage(original_request,destination,deadline)
        require(payload['completed_stream_reference']==stream_ref,'fresh inspection stream reference differs')
        inventory=store.inspect_source(request,inventory_reader=modules['remote'].inventory,deadline=deadline)
        stability=store.compare_source_inventories(controls['before'],inventory)
        value=fresh_stage_record(original_request,stream_ref,inventory,stability)
        body=json_bytes(value);require(len(body)<=request['max_metadata_bytes'],'fresh stage control cap exceeded')
        ref=control_reference(destination/'fresh-source-stage.json',value)
        result=dict(inventory=inventory,stage=dict(store.publish_control(destination/'fresh-source-stage.json',body,ref['sha256'],free_reader=free_reader,deadline=deadline,future_reserve_bytes=request['future_reserve_bytes']),path=ref['path']))
    elif action=='stream':
        require(sink is not None,'binary stream sink required')
        allowed=('expected_index_sha256','expected_archive_sha256','expected_archive_bytes')
        require(set(packet.get('payload',{}))==set(allowed),'exact stream bindings required')
        streamed=store.stream_back(request,destination,sink,deadline=deadline,**packet['payload'])
        if hasattr(sink,'flush'):sink.flush()
        check(deadline)
        index_body=read_bytes(destination/'index.json',META,deadline)
        index_ref=dict(path=str(destination/'index.json'),sha256=streamed['index_sha256'],bytes=len(index_body))
        created,controls=_archive_closure(original_request,destination,index_ref,deadline)
        value=stream_stage_record(original_request,created);body=json_bytes(value)
        require(len(body)<=request['max_metadata_bytes'],'stream stage control cap exceeded')
        store.publish_control(destination/'stream-stage.json',body,hashlib.sha256(body).hexdigest(),free_reader=free_reader,deadline=deadline,future_reserve_bytes=request['future_reserve_bytes'])
        free_reader();return streamed
    elif action=='readcontrol':
        ref=packet['payload']['reference'];path=plain(ref['path'])
        require(path.parent==destination and path.name in ('attempt.json','before.json','after.json','verification.json','origin.json','index.json','preservation-provenance.json','consumer-receipt.json','stream-stage.json','fresh-source-stage.json','writer-process.json'),'closed control name/path required')
        result=dict(body_b64=base64.b64encode(bound_bytes({k:ref[k] for k in ('path','sha256')},deadline,META)).decode())
    else:
        payload=packet['payload'];name=payload['name']
        require(name in ('preservation-provenance.json','consumer-receipt.json'),'closed publication name required')
        body=base64.b64decode(payload['body_b64'],validate=True);require(len(body)<=request['max_metadata_bytes'],'publication residual/control cap exceeded')
        created,controls,fresh,stability,stages=_publication_inputs(original_request,request,destination,deadline)
        value=parse_json(body)
        if name=='preservation-provenance.json':
            _validate_provenance(value,original_request,created,controls['primary'],stability,stages)
        else:
            provenance,prov_ref=_load_stage(destination,'preservation-provenance.json',deadline)
            readback=_validate_provenance(provenance,original_request,created,controls['primary'],stability,stages)
            expected=normalized_receipt(original_request,controls['primary'],controls['before'],fresh,created,primary_reference=created['controls']['primary'],provenance_reference=prov_ref,protocol_stages=stages,readback_identity=readback)
            require(value==expected,'exact normalized consumer receipt closure differs')
        result=dict(store.publish_control(destination/name,body,hashlib.sha256(body).hexdigest(),free_reader=free_reader,deadline=deadline,future_reserve_bytes=request['future_reserve_bytes']),path=str(destination/name))
    free_reader();check(deadline)
    return result


def remote_program(seconds):
    require(type(seconds) is int and 0<seconds<=560,'bounded remote bootstrap wall required')
    # Only a finite wall budget enters argv. All request/authority data travels
    # once over bounded stdin, including inputs larger than platform argv limits.
    return BOOTSTRAP+'''
import signal,time
def bootstrap_alarm(_signum,_frame):raise TimeoutError('outer remote bootstrap deadline expired')
signal.signal(signal.SIGALRM,bootstrap_alarm)
remote_limit=time.monotonic()+'''+str(seconds)+'''
signal.setitimer(signal.ITIMER_REAL,'''+str(seconds)+''')
try:
    packet=read_remote_packet(sys.stdin.buffer)
    remote_limit=min(remote_limit,time.monotonic()+packet['remote_wall_seconds'])
    signal.setitimer(signal.ITIMER_REAL,max(.000001,remote_limit-time.monotonic()))
    request=packet['request']
    m=load_verified_package(request['remote_helper_package_reference'],request,'/',True)['vega_policy_archive.resident_archive']
    signal.setitimer(signal.ITIMER_REAL,0)
    m.remote_main(packet,outer_deadline=remote_limit)
except SystemExit:
    raise
except BaseException as exc:
    sys.stderr.write((type(exc).__name__+': '+str(exc))[:4096]+chr(10))
    raise SystemExit(1)
finally:
    signal.setitimer(signal.ITIMER_REAL,0)
'''


def process_identity(request,*,role='verifier'):
    require(sys.platform=='linux','actual Linux process required')
    interpreter=Path(sys.executable).resolve(strict=True)
    package=request['helper_package_reference']
    require(role in ('writer','verifier'),'closed process role required')
    entry=Path(package['path']).parent/('archive/resident_archive.py' if role=='writer' else 'independent_verify.py')
    return dict(role=role,invocation=dict(argv0=sys.argv[0]),platform=sys.platform,hostname=os.uname().nodename,pid=os.getpid(),ppid=os.getppid(),interpreter=dict(path=str(interpreter),sha256=hashlib.sha256(read_bytes(interpreter,64*MIB,time.monotonic()+5)).hexdigest()),entrypoint=dict(path=str(entry),sha256=hashlib.sha256(read_bytes(entry,131072,time.monotonic()+5)).hexdigest()),source_manifest=package)


def validate_process_pair(writer,verifier):
    require(isinstance(writer,dict) and isinstance(verifier,dict),'full writer/verifier identity required')
    for value in (writer,verifier):
        require(value.get('platform')=='linux' and isinstance(value.get('hostname'),str) and 0<len(value['hostname'])<=255,'honest Linux host required')
        require(type(value.get('pid'))is int and value['pid']>0 and type(value.get('ppid'))is int and value['ppid']>0,'actual process identities required')
        for key in ('interpreter','entrypoint','source_manifest'):
            ref=value.get(key);require(isinstance(ref,dict) and set(ref)=={'path','sha256'} and re.fullmatch('[0-9a-f]{64}',ref['sha256']),'complete process source pin required')
    require(writer['hostname']==verifier['hostname'] and writer['pid']!=verifier['pid'] and writer['ppid']==verifier['pid'],'distinct owned writer child required')
    require(all(writer[key]==verifier[key] for key in ('interpreter','source_manifest')),'writer/verifier source identity differs')
    require(writer.get('role')=='writer' and verifier.get('role')=='verifier' and writer.get('invocation',{}).get('argv0')=='-c','actual writer bootstrap invocation required')
    manifest=parse_json(bound_bytes(writer['source_manifest'],time.monotonic()+5,131072));base=Path(writer['source_manifest']['path']).parent
    for value,name in ((writer,'archive/resident_archive.py'),(verifier,'independent_verify.py')):
        require(value['entrypoint']==dict(path=str(base/name),sha256=manifest['files'][name]['sha256']),'actual process source entry differs')


class LocalTransport:
    """Bounded real child IO. The verifier accepts no value before reaped exit0."""
    def __init__(self,credit_proofs=None):self.credit_proofs=credit_proofs or {}
    def call(self,action,request,*,deadline,consumer=None,**payload):
        check(deadline);seconds=min(560,int(deadline-time.monotonic())-1)
        require(seconds>0,'deadline before writer dispatch')
        packet=dict(action=action,request=request,payload=payload,credit_proofs=self.credit_proofs,remote_wall_seconds=seconds)
        body=json_bytes(packet);require(len(body)<=8*MIB,'writer input cap exceeded')
        result=dependencies()['transport'].run_process([REMOTE_PYTHON,'-B','-I','-S','-u','-c',remote_program(seconds)],body,deadline=deadline,stdout_cap=request['max_archive_bytes'] if action=='stream' else 8*MIB,consumer=consumer)
        if action=='stream':return result['value']
        value=parse_json(result['stdout'])
        if action=='create':require(value['writer_identity']['pid']==result['pid'] and value['writer_identity']['ppid']==os.getpid(),'writer PID differs from observed child')
        return value


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


def remote_main(packet,*,outer_deadline=None):
    previous=None
    try:
        # Trusted bootstrap has already read/parsed/authenticated this same
        # packet once. Never reread stdin or reconstruct a different request.
        outer_deadline=min(time.monotonic()+560,outer_deadline) if outer_deadline is not None else time.monotonic()+560
        previous=_watchdog(outer_deadline)
        deadline=min(outer_deadline,remote_deadline(packet['remote_wall_seconds']))
        signal.setitimer(signal.ITIMER_REAL,max(.000001,deadline-time.monotonic()))
        result=remote_dispatch(packet,deadline=deadline,sink=sys.stdout.buffer)
        if packet['action']!='stream':
            data=json_bytes(result);require(len(data)<=8*MIB,'remote stdout JSON cap exceeded');sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
    except BaseException as exc:
        sys.stderr.write((type(exc).__name__+': '+str(exc))[:4096]+'\n');raise SystemExit(1)
    finally:
        if previous is not None:_unwatch(previous)


def run(request_path,output):
    """Public Linux verifier coordinator. No archive or raw extraction is written locally."""
    started=time.monotonic();deadline=started+600;previous=_watchdog(deadline);guard=None
    try:
        request_path=plain(str(request_path));body=read_bytes(request_path,EVIDENCE,deadline);public=parse_json(body)
        require(public.get('schema')==SCHEMA,'resident request required')
        package_ref=dict(public['helper_package_reference'])
        if not Path(package_ref['path']).is_absolute():package_ref['path']=str(request_path.parent/package_ref['path'])
        loaded=verify_package(package_ref,authority_request=public,base=request_path.parent)
        # Re-enter the verified source module. An imported unverified coordinator
        # cannot cause a dependency to load before this complete package check.
        implementation=loaded['vega_policy_archive.resident_archive']
        require(plain(implementation.__file__).parent==plain(package_ref['path']).parent/'archive','entry source/package directory differs')
        return implementation._run_verified(request_path,output,body,started,deadline)
    finally:_unwatch(previous)


def _run_verified(request_path,output,body,started,deadline):
    public=parse_json(body)
    seconds=public.get('wall_seconds');require(type(seconds) is int and 0<seconds<=600,'bounded positive wall_seconds required')
    deadline=min(deadline,started+seconds);signal.setitimer(signal.ITIMER_REAL,max(.000001,deadline-time.monotonic()))
    if public.get('kind')=='CONTROL_ONLY':
        from . import control_preservation
        return control_preservation.preserve_verified(public,request_path,output,deadline=deadline)
    request=validate(public,request_path,deadline)
    require(sys.platform=='linux','actual independent Linux readback required')
    require(plain(str(output))==Path(CONTROL_ROOT)==plain(request['control_root']),'explicit admitted Vega control root required')
    modules=dependencies();guard=modules['capacity'].ControlRoot(output,free_reader=quota_headroom,deadline=deadline);guard.check()
    # Snapshot every original binding and request before any remote effect.
    references=[dict(path=str(request_path),sha256=hashlib.sha256(body).hexdigest())]
    references += [dict(path=x['path'],sha256=x['sha256']) for x in request['_bound_extra']]
    references += [{k:request['_bound_evidence'][k] for k in ('path','sha256')}]
    references += [public[k] for k in ('helper_package_reference','capacity_contract_reference','capacity_ledger_reference')]
    def recheck():
        check(deadline)
        for ref in references:bound_bytes(ref,deadline,base=request_path.parent)
        # Includes the entry source and all helper bytes, not just the manifest.
        verify_package(dict(path=str(Path(request['helper_package_reference']['path']) if Path(request['helper_package_reference']['path']).is_absolute() else request_path.parent/request['helper_package_reference']['path']),sha256=request['helper_package_reference']['sha256']),authority_request=public,base=request_path.parent)
    recheck()
    request['_ledger_body_b64']=base64.b64encode(bound_bytes(public['capacity_ledger_reference'],deadline,base=request_path.parent)).decode()
    proofs={}
    for _,_,item in modules['capacity'].retained_objects(request['_ledger']):
        raw=bound_bytes(item['proof'],deadline,MIB,base=request_path.parent)
        proofs[item['proof']['sha256']]=base64.b64encode(raw).decode()
    transport=LocalTransport(proofs)
    try:return coordinate(request,transport,guard,deadline=deadline,recheck=recheck)
    except BaseException as exc:
        try:guard.write(request['policy_id']+'-'+request['job_id']+'-error.json',json_bytes(dict(schema='z1_vega_resident_failure_v1',status='FAILED_NO_RETRY',error_type=type(exc).__name__,error=str(exc)[:4096],request_identity_sha256=request_identity(request))))
        except BaseException:pass
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.request,args.output);sys.stdout.write(json_bytes(result).decode()+'\n')

if __name__=='__main__':main()
