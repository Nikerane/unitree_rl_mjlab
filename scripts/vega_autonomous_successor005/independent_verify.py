#!/usr/bin/env python3
"""Source-pinned Linux independent verifier. Caller must observe actual exit0."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
MODE='VEGA_RESIDENT_FULL_ARCHIVE_VEGA_PROCESS_V1'


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

def preserve(request_path: str, output: str):
    path=Path(request_path)
    if not path.is_absolute() or '..' in path.parts:raise ValueError('absolute request required')
    # The bootstrap performs bounded no-follow authority/source reads before helpers.
    import os,stat
    for parent in reversed(path.parents):
        if not stat.S_ISDIR(parent.lstat().st_mode):raise ValueError('request parent symlink')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size>8388608:raise ValueError('request cap/type')
        body=stream.read(8388609);after=os.fstat(stream.fileno())
    fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    if len(body)>8388608 or tuple(getattr(before,k) for k in fields)!=tuple(getattr(after,k) for k in fields):raise ValueError('request drift')
    packet=read_remote_packet(__import__('io').BytesIO(b'{"remote_wall_seconds":560,"request":'+body+b'}'))
    request=packet['request'];loaded=load_verified_package(request['helper_package_reference'],request,str(path.parent))
    result=loaded['vega_policy_archive.resident_archive'].run(str(path),output)
    return result.get('independent_backup',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=preserve(args.request,args.output)
    if set(result)!={'path','sha256'}:raise ValueError('full receipt reference required')
    sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        sys.stderr.write((type(exc).__name__+': '+str(exc))[:4096]+'\n');raise SystemExit(1)
