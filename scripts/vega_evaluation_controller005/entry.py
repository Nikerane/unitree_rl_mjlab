#!/usr/bin/env python3
"""Isolated stdlib authority-before-source entry. Caller pins THIS entry first.

Task2 intentionally cannot activate a physical adapter. Task3 extends the exact
source inventory and wires its literal adapter only after independent review.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import types

SOURCES=('journal.py','scheduler.py','coordinator.py','entry.py','controller.sbatch')
CPU_ROOT='/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002'
CONTROL_ROOT=CPU_ROOT+'/admission-staged-integration002/autonomous005'
PRESERVATION_SHA='be0dca5fe9175b253b5c297ce3e67b72012678bee99389cea42d71e114aae47b'

class AdmissionError(RuntimeError): pass

def require(condition,message):
    if not condition: raise AdmissionError(message)

def read(path,cap=1048576):
    path=Path(path)
    require(path.is_absolute(),'absolute path required')
    for parent in [path]+list(path.parents): require(not parent.is_symlink(),'symlink reference')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        st=os.fstat(fd)
        require(stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_size<=cap,'bounded regular reference required')
        data=bytearray()
        while True:
            block=os.read(fd,min(65536,cap+1-len(data)))
            if not block: break
            data.extend(block); require(len(data)<=cap,'reference cap')
        after=os.fstat(fd)
        require((st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'reference changed')
        return bytes(data)
    finally: os.close(fd)

def pairs(values):
    result={}
    for k,v in values:
        require(k not in result,'duplicate JSON key'); result[k]=v
    return result

def parse(data):
    return json.loads(data,object_pairs_hook=pairs,parse_constant=lambda _:require(False,'nonfinite JSON'))

def referenced(reference):
    require(isinstance(reference,dict) and set(reference)=={'path','sha256'},'reference schema')
    require(isinstance(reference['sha256'],str) and re.fullmatch('[0-9a-f]{64}',reference['sha256']),'SHA256 required')
    data=read(reference['path'])
    require(hashlib.sha256(data).hexdigest()==reference['sha256'],'reference SHA256 mismatch')
    return data

def verify_source(manifest_path,manifest_sha):
    manifest_reference={'path':manifest_path,'sha256':manifest_sha}
    manifest=parse(referenced(manifest_reference))
    require(isinstance(manifest,dict) and set(manifest)=={'schema','sources'} and manifest['schema']=='z1_vega_controller_source_v1','manifest schema')
    require(isinstance(manifest['sources'],dict) and set(manifest['sources'])==set(SOURCES),'closed source inventory')
    root=Path(manifest_path).parent
    require(Path(__file__).absolute()==root/'entry.py','entry outside manifest directory')
    # Staged directory is SOURCE ONLY. Tests/reports are never runtime inputs.
    require({p.name for p in root.iterdir()}==set(SOURCES)|{'source-manifest.json'},'extra source/control in staged directory')
    contents={}
    for name in SOURCES:
        record=manifest['sources'][name]
        require(set(record)=={'sha256','bytes'} and type(record['bytes']) is int,'source record schema')
        raw=referenced({'path':str(root/name),'sha256':record['sha256']})
        require(len(raw)==record['bytes'],'source byte count')
        if name.endswith('.py'): compile(raw,str(root/name),'exec')
        contents[name]=raw
    return contents

def authenticate(config_path,delegation_path,delegation_sha,manifest_path,manifest_sha,*,read_only=False):
    require(sys.platform=='linux','activation/reconciliation requires honest Linux')
    require(config_path==CONTROL_ROOT+'/controller-config.json','closed config path')
    require(delegation_path==CONTROL_ROOT+'/controller-delegation.json','closed delegation path')
    delegation=parse(referenced({'path':delegation_path,'sha256':delegation_sha}))
    require(set(delegation)=={'schema','config','source_manifest','nonce','permissions'},'delegation fields')
    require(delegation['schema']=='z1_vega_controller_delegation_v1','delegation schema')
    require(delegation['permissions']==['remaining-five','science-twelve','preserve-controls','publish-authority'],'closed delegation permissions')
    require(delegation['config']['path']==config_path,'delegation config binding')
    expected_source={'path':manifest_path,'sha256':manifest_sha}
    require(delegation['source_manifest']==expected_source,'delegation source binding')
    config_data=referenced(delegation['config']); config=parse(config_data)
    require(set(config)=={'schema','nonce','source_manifest','preservation_manifest','controller_job_id','uid','root','log_root','batches'},'config fields')
    require(config['schema']=='z1_vega_controller_config_v1','config schema')
    require(isinstance(config['nonce'],str) and re.fullmatch('[0-9a-f]{32,64}',config['nonce']) and config['nonce']==delegation['nonce'],'delegation nonce')
    require(config['source_manifest']==expected_source,'config source binding')
    require(config['preservation_manifest']['sha256']==PRESERVATION_SHA,'closed preservation source')
    referenced(config['preservation_manifest'])
    require(config['root']==CONTROL_ROOT+'/controller-state005','closed state root')
    require(config['log_root']==CPU_ROOT+'/evidence-autonomous005','closed provenance root')
    require(re.fullmatch('[1-9][0-9]*',config['controller_job_id']) and (read_only or config['controller_job_id']==os.environ.get('SLURM_JOB_ID')),'Slurm controller binding')
    require(config['uid']==str(os.getuid()),'controller UID binding')
    require(set(config['batches'])=={'remaining','science'},'closed phase batches')
    for phase,reference in config['batches'].items():
        require(reference['path']==CONTROL_ROOT+'/controller-inputs/'+phase+'.sbatch','closed batch path')
        referenced(reference)
    return config,{'nonce':config['nonce'],'controller_job_id':config['controller_job_id'],'uid':config['uid'],
                   'source_sha256':manifest_sha,'config_sha256':hashlib.sha256(config_data).hexdigest(),'log_root':config['log_root']}

def load_sources(contents,root):
    # Verified byte compilation only; no sys.path import or bytecode fallback.
    for name in ('journal','scheduler','coordinator'):
        require(name not in sys.modules,'ambient module collision')
        module=types.ModuleType(name); module.__file__=str(root/(name+'.py'))
        sys.modules[name]=module
        exec(compile(contents[name+'.py'],module.__file__,'exec'),module.__dict__)
    return sys.modules['coordinator']

def main():
    parser=argparse.ArgumentParser()
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check-source',metavar='MANIFEST')
    action.add_argument('--reconcile',action='store_true')
    action.add_argument('--activate',action='store_true')
    parser.add_argument('--manifest'); parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--config'); parser.add_argument('--delegation'); parser.add_argument('--delegation-sha256')
    args=parser.parse_args()
    manifest=args.check_source or args.manifest
    require(manifest is not None,'manifest required')
    # Authority precedes loading candidate source. --check-source executes none.
    if args.check_source:
        verify_source(manifest,args.manifest_sha256)
        print(json.dumps({'status':'SOURCE_VERIFIED'})); return
    config,context=authenticate(args.config,args.delegation,args.delegation_sha256,manifest,args.manifest_sha256,read_only=args.reconcile)
    contents=verify_source(manifest,args.manifest_sha256)
    module=load_sources(contents,Path(manifest).parent)
    if args.reconcile:
        print(json.dumps(module.reconcile(Path(config['root'])),sort_keys=True)); return
    raise AdmissionError('HOLD: Task3 closed physical/review adapter not yet installed; activation disabled')

if __name__=='__main__':
    try: main()
    except (AdmissionError,OSError,ValueError,TypeError,KeyError) as exc:
        print('HOLD: '+str(exc),file=sys.stderr); sys.exit(75)
