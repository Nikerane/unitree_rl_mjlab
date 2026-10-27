"""Acyclic small-control archives through the full store and nonseek verifier.

This source route authenticates root preservation-authority.json before loading
candidate source. It has no CPU release dependency and grants no rollout power.
"""
from __future__ import annotations
import base64
import hashlib
import os
from pathlib import Path
import re
import stat
import time


def _modules():
    from vega_policy_archive import resident_archive as a, reservation as r
    return a,r,a.dependencies()


def validate(request,*,deadline):
    a,r,m=_modules();a.require(request.get('kind')=='CONTROL_ONLY' and request.get('backend_mode')==a.MODE,'explicit control-only kind/mode required')
    a.require(not any(key.startswith('_') for key in request),'private request fields forbidden')
    a.require(not any(key in request for key in ('cpu_supervisor_release','remote_supervisor_release','cpu_supervisor_manifest','policy','scope','bank','job_id')),'control-only cannot claim policy/release authority')
    name=request.get('control_id');a.require(isinstance(name,str) and re.fullmatch('[a-z][a-z0-9-]{0,63}',name),'closed control id required')
    a.require(request['control_root']==r.CONTROL_ROOT,'exact control root required')
    base=Path(r.CONTROL_ROOT)/'input-bundles'/name
    a.require(request['source_root']==str(base/'controls') and request['private_root']==str(base/'private'),'closed control input roots required')
    destination=Path(r.BACKUP_ROOT)/'control-only'/name
    a.require(request['remote_archive_destination']==str(destination),'closed generic backup destination required')
    m['store']._bounds(request)
    a.require(request['max_source_bytes']<=r.SHARED['controls'] and request['max_archive_bytes']<=r.SHARED['controls'],'small control caps required')
    a.require(type(request.get('wall_seconds'))is int and 0<request['wall_seconds']<=600,'bounded wall seconds required')
    ledger=a.parse_json(a.bound_bytes(request['capacity_ledger_reference'],deadline))
    R=r._ledger_budget(ledger)
    a.require(not any(slot['retained'][role] for slot in ledger['slots'] for role in r.CAPS),'generic bootstrap uses conservative zero credit ledger')
    return destination,ledger,R


def inventory(request):
    a,r,m=_modules();deadline=time.monotonic()+request['wall_seconds'];nodes=m['store']._walk(request,deadline)
    for row in nodes.values():
        if row['type']=='file':row['sha256']=hashlib.sha256(a.read_bytes(row['source_path'],request['max_source_bytes'],deadline)).hexdigest()
    return dict(nodes=nodes,kind='CONTROL_ONLY',control_id=request['control_id'],source_root=request['source_root'],private_root=request['private_root'],job={})


def _usage(deadline):
    a,r,m=_modules();total=0;count=0;seen=set()
    def walk(path):
        nonlocal total,count
        a.check(deadline)
        for parent in reversed(path.parents):
            try:info=parent.lstat()
            except FileNotFoundError:continue
            a.require(stat.S_ISDIR(info.st_mode),'generic usage parent symlink forbidden')
        try:info=path.lstat()
        except FileNotFoundError:return
        count+=1;a.require(count<=8192,'generic node cap exceeded')
        if stat.S_ISDIR(info.st_mode):
            fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:
                with os.scandir(fd) as entries:
                    for entry in entries:walk(path/entry.name)
            finally:os.close(fd)
        else:
            a.require(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'generic control symlink/special/hardlink forbidden')
            inode=(info.st_dev,info.st_ino);a.require(inode not in seen,'generic inode overlap');seen.add(inode)
            total+=info.st_size;a.require(total<=r.SHARED['controls'],'shared controls64MiB exhausted')
    walk(Path(r.CONTROL_ROOT));walk(Path(r.BACKUP_ROOT)/'control-only')
    return total


def _refs(destination,deadline):
    a,r,m=_modules();index=a.parse_json(a.read_bytes(destination/'index.json',a.META,deadline))
    refs=index['controls'];controls={key:a.parse_json(a.bound_bytes({k:ref[k] for k in ('path','sha256')},deadline,a.META)) for key,ref in refs.items()}
    a.require(set(refs)=={'attempt','before','after','primary','origin'},'complete generic metadata required')
    names=dict(attempt='attempt.json',before='before.json',after='after.json',primary='verification.json',origin='origin.json')
    for key,ref in refs.items():a.require(ref['path']==str(destination/names[key]),'foreign generic control metadata')
    m['store'].compare_source_inventories(controls['before'],controls['after'])
    primary=controls['primary'];a.require(primary['origin_map']==controls['origin'],'generic primary origin differs')
    a.require(index['archive_sha256']==primary['archive_sha256'] and index['archive_bytes']==primary['archive_bytes'],'generic archive index differs')
    writer=a.parse_json(a.read_bytes(destination/'writer-process.json',a.META,deadline))
    return index,controls,writer


def dispatch(packet,*,deadline,sink=None):
    a,r,m=_modules();request=packet['request'];destination,ledger,R=validate(request,deadline=deadline)
    protected=max(r.IMMUTABLE,R-r.SHARED['controls'])
    effective=dict(request,future_reserve_bytes=protected-m['capacity'].FLOOR,remote_future_reserve_bytes=protected-m['capacity'].FLOOR)
    def free_reader():
        used=_usage(deadline);actual=a.quota_headroom()
        a.require(actual>=max(protected,R-used),'generic current quota below reservation')
        return min(actual,protected+r.SHARED['controls']-used)
    free_reader();action=packet['action'];payload=packet.get('payload',{})
    if action=='create':
        result=m['store'].create_archive(effective,destination,inventory_reader=inventory,free_reader=free_reader,deadline=deadline)
        result['writer_identity']=a.process_identity(request,role='writer');body=a.json_bytes(result['writer_identity'])
        m['store'].publish_control(destination/'writer-process.json',body,hashlib.sha256(body).hexdigest(),free_reader=free_reader,deadline=deadline,future_reserve_bytes=effective['future_reserve_bytes'])
        return result
    if action=='readcontrol':
        ref=payload['reference'];path=a.plain(ref['path'])
        a.require(path.parent==destination and path.name in ('attempt.json','before.json','after.json','verification.json','origin.json','index.json','writer-process.json','stream-stage.json','fresh-source-stage.json','preservation-provenance.json','consumer-receipt.json'),'closed generic control read required')
        return dict(body_b64=base64.b64encode(a.bound_bytes({k:ref[k] for k in ('path','sha256')},deadline,a.META)).decode())
    index,controls,writer=_refs(destination,deadline)
    if action=='stream':
        a.require(sink is not None,'generic stream sink required')
        result=m['store'].stream_back(effective,destination,sink,deadline=deadline,**payload)
        sink.flush();stage=dict(schema='z1_vega_control_stream_stage_v1',request_sha256=a.request_identity(request),archive_sha256=result['archive_sha256'],archive_bytes=result['archive_bytes'])
        body=a.json_bytes(stage);m['store'].publish_control(destination/'stream-stage.json',body,hashlib.sha256(body).hexdigest(),free_reader=free_reader,deadline=deadline,future_reserve_bytes=effective['future_reserve_bytes'])
        free_reader();return result
    stream=a.parse_json(a.read_bytes(destination/'stream-stage.json',a.META,deadline))
    a.require(stream==dict(schema='z1_vega_control_stream_stage_v1',request_sha256=a.request_identity(request),archive_sha256=index['archive_sha256'],archive_bytes=index['archive_bytes']),'generic completed stream stage required')
    if action=='inspect':
        current=m['store'].inspect_source(effective,inventory_reader=inventory,deadline=deadline)
        stability=m['store'].compare_source_inventories(controls['before'],current)
        value=dict(schema='z1_vega_control_fresh_stage_v1',request_sha256=a.request_identity(request),inventory=current,source_stability=stability)
        body=a.json_bytes(value);ref=a.control_reference(destination/'fresh-source-stage.json',value)
        m['store'].publish_control(destination/'fresh-source-stage.json',body,ref['sha256'],free_reader=free_reader,deadline=deadline,future_reserve_bytes=effective['future_reserve_bytes'])
        return dict(inventory=current,stage=ref)
    a.require(action=='publish','closed generic action required')
    fresh=a.parse_json(a.read_bytes(destination/'fresh-source-stage.json',a.META,deadline))
    m['store'].compare_source_inventories(fresh['inventory'],m['store'].inspect_source(effective,inventory_reader=inventory,deadline=deadline))
    name=payload['name'];a.require(name in ('preservation-provenance.json','consumer-receipt.json'),'closed generic publication required')
    body=base64.b64decode(payload['body_b64'],validate=True);value=a.parse_json(body)
    if name=='preservation-provenance.json':
        a.validate_process_pair(writer,value.get('verifier_identity'))
        expected=_provenance(request,controls['primary'],writer,value['verifier_identity'],fresh,stream)
    else:
        provenance=a.parse_json(a.read_bytes(destination/'preservation-provenance.json',a.META,deadline));a.validate_process_pair(writer,provenance['verifier_identity'])
        a.require(provenance==_provenance(request,controls['primary'],writer,provenance['verifier_identity'],fresh,stream),'generic provenance exact closure differs')
        expected=_receipt(request,destination,controls['primary'],a.control_reference(destination/'preservation-provenance.json',provenance))
    a.require(value==expected,'generic exact publication closure differs')
    return dict(m['store'].publish_control(destination/name,body,hashlib.sha256(body).hexdigest(),free_reader=free_reader,deadline=deadline,future_reserve_bytes=effective['future_reserve_bytes']),path=str(destination/name))


def _provenance(request,primary,writer,verifier,fresh,stream):
    a,r,m=_modules()
    return dict(schema='z1_vega_control_preservation_provenance_v1',kind='CONTROL_ONLY',status='PASS',request=request,request_sha256=a.request_identity(request),writer_identity=writer,verifier_identity=verifier,archive_sha256=primary['archive_sha256'],archive_bytes=primary['archive_bytes'],create_transport_exit=0,stream_transport_exit=0,complete_eof=True,stream_stage=stream,fresh_stage=fresh,source_manifest=request['helper_package_reference'])


def _receipt(request,destination,primary,provenance):
    a,r,m=_modules()
    return dict(schema='z1_vega_control_archive_receipt_v1',kind='CONTROL_ONLY',status='PASS',backend_mode=a.MODE,independent_readback='PASS',control_id=request['control_id'],archive_path=str(destination/'terminal-policy-full.tar.gz'),archive_sha256=primary['archive_sha256'],archive_bytes=primary['archive_bytes'],node_metadata=primary['origin_map'],source_manifest=request['helper_package_reference'],provenance=provenance,technical_qualification=False,scientific_qualification=False,raw_tree_extracted=False)


def preserve_verified(request,request_path,output,*,deadline):
    a,r,m=_modules();destination,ledger,R=validate(request,deadline=deadline)
    a.require(__import__('sys').platform=='linux','actual Linux verifier required');a.require(str(output)==r.CONTROL_ROOT,'exact generic output root required')
    body=a.read_bytes(request_path,a.EVIDENCE,deadline);source_ref=request['helper_package_reference']
    def recheck():
        a.require(a.parse_json(a.read_bytes(request_path,a.EVIDENCE,deadline))==request,'generic request drift')
        a.verify_package(source_ref,authority_request=request,base=request_path.parent)
        a.bound_bytes(request['capacity_ledger_reference'],deadline)
    guard=m['capacity'].ControlRoot(output,free_reader=a.quota_headroom,deadline=deadline)
    with guard.lease():
        recheck();transport=a.LocalTransport();created=transport.call('create',request,deadline=deadline)
        a.require(created['destination']==str(destination),'generic writer destination differs');a.validate_process_pair(created['writer_identity'],a.process_identity(request))
        refs=created['controls'];primary=a.control_read(transport,request,refs['primary'],deadline);before=a.control_read(transport,request,refs['before'],deadline)
        independent=transport.call('stream',request,deadline=deadline,consumer=lambda stream:m['reader'].verify_stream(stream,before,max_archive_bytes=request['max_archive_bytes'],max_source_bytes=request['max_source_bytes'],max_metadata_bytes=request['max_metadata_bytes'],max_evidence_bytes=request['max_evidence_bytes'],application_names=[],deadline=deadline),expected_index_sha256=refs['index']['sha256'],expected_archive_sha256=created['archive_sha256'],expected_archive_bytes=created['archive_bytes'])
        a.require(independent==primary,'generic independent stream differs')
        inspected=transport.call('inspect',request,deadline=deadline)
        fresh=a.control_read(transport,request,inspected['stage'],deadline)
        stream=dict(schema='z1_vega_control_stream_stage_v1',request_sha256=a.request_identity(request),archive_sha256=primary['archive_sha256'],archive_bytes=primary['archive_bytes'])
        a.require(a.control_read(transport,request,a.control_reference(destination/'stream-stage.json',stream),deadline)==stream,'generic stream stage differs')
        recheck();provenance=_provenance(request,primary,created['writer_identity'],a.process_identity(request),fresh,stream)
        prov=a.publish(transport,request,'preservation-provenance.json',provenance,deadline)
        full=_receipt(request,destination,primary,prov);reference=a.publish(transport,request,'consumer-receipt.json',full,deadline)
        recheck();return {k:reference[k] for k in ('path','sha256')}
