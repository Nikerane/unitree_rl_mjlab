"""Finite031 CLI; direct -I -S entry verifies source before package imports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def bootstrap():
    if __package__:
        return
    # Isolated direct-file mode trusts only the root-pinned manifest digest.
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--manifest',required=True);parser.add_argument('--manifest-sha',required=True)
    args,_=parser.parse_known_args()
    raw=Path(args.manifest).read_bytes()
    value=json.loads(raw)
    canonical=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest()!=args.manifest_sha:raise ValueError('bootstrap manifest identity differs')
    root=Path(value['source_root'])
    if Path(__file__).resolve()!=root/'scripts/z1_depth_campaign031/__main__.py':raise ValueError('bootstrap source origin differs')
    for name,sha in value['source_files'].items():
        path=root/name
        if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():raise ValueError('bootstrap nonsymlink source required')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('bootstrap source bytes differ: '+name)
    sys.path.insert(0,str(root))
    globals()['__package__']='scripts.z1_depth_campaign031'


bootstrap()


def main():
    from .files import digest,need,read_json,write_once
    from .manifest import prepare,validate,verify_package
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest');parser.add_argument('--manifest-sha')
    sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--workspace',required=True);prep.add_argument('--destination',required=True);prep.add_argument('--evidence',required=True)
    for name in ('verify','submission-plan','qualify','controller','slot','evaluation'):
        cmd=sub.add_parser(name)
        if name in ('qualify','controller','slot','evaluation'):cmd.add_argument('--grant',required=True)
        if name in ('slot','evaluation'):cmd.add_argument('--index',type=int,required=True)
    child=sub.add_parser('child');child.add_argument('--intent',required=True);child.add_argument('--intent-sha',required=True)
    save=sub.add_parser('archive');save.add_argument('--source',required=True);save.add_argument('--destination',required=True);save.add_argument('--identity',required=True);save.add_argument('--cap',type=int,required=True);save.add_argument('--receipt',required=True)
    join=sub.add_parser('join');join.add_argument('--inputs',required=True);join.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.action=='prepare':
        result=prepare(args.workspace,args.destination,args.evidence);print(digest(result));return
    need(args.manifest and args.manifest_sha,'pinned manifest required')
    manifest=read_json(args.manifest);need(digest(manifest)==args.manifest_sha,'manifest digest differs');validate(manifest)
    from . import worker
    if args.action=='verify':verify_package(manifest);result={'status':'VERIFIED_CONTENT_ONLY'}
    elif args.action=='submission-plan':result=worker.submission_plan(manifest)
    elif args.action=='child':result=worker.execute_child(args.intent,args.intent_sha)
    elif args.action=='archive':
        from .preservation import archive,verify_ssd
        verify_ssd(manifest,args.destination)
        result=archive(args.source,args.destination,identity=args.identity,cap=args.cap)
        write_once(args.receipt,result)
    elif args.action=='join':
        from .preservation import final_join
        inputs=read_json(args.inputs);result=final_join(manifest,inputs['training'],inputs['evaluations'],inputs['backups']);write_once(args.output,result)
    else:
        grant=read_json(args.grant)
        if args.action=='qualify':result=worker.qualify(manifest,grant)
        elif args.action=='controller':result=worker.controller(manifest,grant)
        elif args.action=='slot':result=worker.run_slot(manifest,grant,args.index)
        else:result=worker.run_evaluation(manifest,grant,args.index)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
