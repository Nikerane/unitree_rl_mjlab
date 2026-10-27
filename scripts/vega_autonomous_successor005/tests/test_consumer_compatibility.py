"""Actual unchanged565 consumer +86c98160 sealer, real synthetic32x881 NPZ.

Only source/root/platform/job inventory is translated in disposable fixtures.
No scheduler or model/environment construction is invoked.
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from fixture_support import FROZEN,OLD,body,generic,load,sha,write

class ConsumerTests(unittest.TestCase):
    def test_real_npz_full_receipt_accepted_by_actual_unchanged_sealer_consumer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp).resolve();root,generic_request,ref=generic(base)
            entry=load(root/'independent_verify.py','fixture_verified_entry')
            loaded=entry.load_verified_package(generic_request['helper_package_reference'],generic_request,str(Path(ref['path']).parent));a=loaded['vega_policy_archive.resident_archive'];m=a.dependencies()
            sys.path.insert(0,str(OLD))
            old_store=load(OLD/'test_resident_store.py','fixture_old_store')
            old_pipeline=load(OLD/'test_resident_archive.py','fixture_old_pipeline')
            fixture=old_store.ResidentStoreTests('test_complete_archive_preserves_originals');fixture.setUp();self.addCleanup(fixture.doCleanups)
            fixture.success_records();request=fixture.request
            request['remote_supervisor_release']=dict(path='/fixture/release.json',sha256='abc')
            request.update(shared_source=dict(archive_sha256='fixture-only'),backend_mode=a.MODE,helper_package_reference=generic_request['helper_package_reference'],remote_helper_package_reference=generic_request['helper_package_reference'],capacity_ledger_reference=generic_request['capacity_ledger_reference'])
            captured=FROZEN/'historical-fixtures/evaluation-preparation/scanner-repair-v001/seal-tools/test_seal_pure.py'
            factory=load(captured,'actual_npz_fixture',FROZEN/'seal-tools/test_seal_pure.py')
            seal_parent=base/'seal';seal_parent.mkdir();seal_request=factory.fixture(seal_parent,miss=True)
            raw=Path(seal_request['raw_root']);policy=json.loads((raw/'intent.json').read_bytes())['policy']
            fixture.raw=raw;request.update(source_root=str(raw),policy=policy,policy_id=policy['policy_id']);fixture.destination=fixture.backup/policy['policy_id']/raw.name/('job-'+request['job_id']);fixture.destination.parent.mkdir(parents=True,exist_ok=True)
            for name in ('supervisor-terminal.json','intent.json','worker-result.json'):
                value=json.loads((raw/name).read_bytes())
                if name=='supervisor-terminal.json':value['allocation']={'SLURM_JOB_ID':request['job_id']}
                if name=='intent.json':value.update(output_root=str(raw),supervisor_context={'release_sha256':'abc'})
                if name=='worker-result.json':value['status']='RAW_REVIEW_REQUIRED'
                (raw/name).write_bytes(body(value))
            request['_bound_observation']=dict(binding=dict(records={name:dict(value=json.loads((raw/name).read_bytes()),sha256=sha(raw/name)) for name in ('supervisor-terminal.json','intent.json','worker-result.json')}))
            seal_request['intent']['sha256']=sha(raw/'intent.json');tree=factory.w.tree_digest(factory.w.inventory(raw))
            for role in ('raw','runtime','parity'):
                item=seal_request[role+'_review'];value=json.loads(Path(item['path']).read_bytes());value.update(raw_tree_sha256=tree,intent_sha256=seal_request['intent']['sha256']);Path(item['path']).write_bytes(body(value));item['sha256']=sha(item['path'])
            guard=m['capacity'].ControlRoot(request.get('control_root',generic_request['control_root']),free_reader=lambda:10**12,deadline=fixture.deadline)
            transport=old_pipeline.LocalTransport(fixture)
            # Source-pinned real writer child; only IO roots/scheduler inventory
            # are fixture inputs. Stream is a real bounded process pipe as well.
            original_call=transport.call
            def child(action,payload,consumer=None):
                code="""import json,sys,time,types,hashlib,os
from pathlib import Path
packet=json.loads(sys.stdin.buffer.read(8388609));path=Path(packet['entry']);e=types.ModuleType('entry');e.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),e.__dict__)
loaded=e.load_verified_package(packet['authority']['helper_package_reference'],packet['authority'],str(path.parent));a=loaded['vega_policy_archive.resident_archive'];m=a.dependencies();s=m['store'];r=packet['request'];d=Path(packet['destination']);s.BACKUP_ROOT=Path(packet['backup']);m['contract'].validate_paths=lambda r:{'policy-tree':Path(r['source_root']),'private-tree':Path(r['private_root'])};deadline=time.monotonic()+25
def inventory(r):
 nodes=s._walk(r,deadline)
 for row in nodes.values():
  if row['type']=='file':row['sha256']=hashlib.sha256(Path(row['source_path']).read_bytes()).hexdigest()
 c=r['scheduler_identity_capture'];identity=m['contract'].accounting_identity(r['job_id'],c['accounting_job_id'],r['job_id'],c['capture'])
 return dict(nodes=nodes,policy_id=r['policy_id'],scope=r['scope'],source_root=r['source_root'],private_root=r['private_root'],job=dict(job_id=r['job_id'],terminal=True,state='COMPLETED',exit_code='0:0',identity=identity))
if packet['action']=='create':
 result=s.create_archive(r,d,inventory_reader=inventory,free_reader=lambda:10**12,deadline=deadline);result['writer_identity']=a.process_identity(r);b=a.json_bytes(result['writer_identity']);s.publish_control(d/'writer-process.json',b,hashlib.sha256(b).hexdigest(),free_reader=lambda:10**12,deadline=deadline);sys.stdout.buffer.write(a.json_bytes(result))
else:
 s.stream_back(r,d,sys.stdout.buffer,deadline=deadline,**packet['payload']);sys.stdout.buffer.flush();idx=(d/'index.json').read_bytes();created,_=a._archive_closure(r,d,dict(path=str(d/'index.json'),sha256=hashlib.sha256(idx).hexdigest(),bytes=len(idx)),deadline);b=a.json_bytes(a.stream_stage_record(r,created));s.publish_control(d/'stream-stage.json',b,hashlib.sha256(b).hexdigest(),free_reader=lambda:10**12,deadline=deadline)
"""
                packet=dict(action=action,entry=str(root/'independent_verify.py'),authority=generic_request,request=request,destination=str(fixture.destination),backup=str(fixture.backup),payload=payload)
                result=m['transport'].run_process([sys.executable,'-B','-I','-S','-c',code],body(packet),deadline=fixture.deadline,stdout_cap=request['max_archive_bytes'],consumer=consumer)
                return result['value'] if consumer else json.loads(result['stdout'])
            def call(action,request,*,deadline,consumer=None,**payload):
                if action in ('create','stream'):return child(action,payload,consumer)
                return original_call(action,request,deadline=deadline,consumer=consumer,**payload)
            transport.call=call
            result=a.coordinate(request,transport,guard,deadline=fixture.deadline,recheck=lambda:None)
            seal_request['independent_backup']=result['independent_backup'];seal_ref=factory.w.build(seal_request)
            seal=json.loads(Path(seal_ref['path']).read_bytes());self.assertEqual(seal['status'],'QUALIFIED');self.assertEqual(seal['denominators']['contact_801'],31);self.assertEqual(seal['denominators']['genuine_no_contact'],1);self.assertTrue(seal['unsafe_allowed_for_technical_qualification'])
            consumer=factory.w.load_consumer();self.assertEqual(sha(consumer.__file__),'565bb43027d8c9a98505603985d7a95b8f54c093c51e6cbcb2ba947f1507c4fd')
            self.assertEqual(sha(factory.w.__file__),'86c98160bd9bda67ca54d4600a73c1280ee38917304e200ee0b348e6cf1c29e0')
            consumer.verify_qualification(json.loads((raw/'intent.json').read_bytes()),{k:seal_ref[k] for k in ('path','sha256')})
    def test_actual_phase_gate_selected_slot_and_reservation_index_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp).resolve();root,request,ref=generic(base);entry=load(root/'independent_verify.py','phase_entry');loaded=entry.load_verified_package(request['helper_package_reference'],request,str(Path(ref['path']).parent));a=loaded['vega_policy_archive.resident_archive'];m=a.dependencies()
            phase=load(FROZEN/'tests/test_phase_controls.py','actual_phase_fixture')
            with phase.Fixture('remaining_own32',phase.REMAINING) as f:
                f.qualify_first();binary=f.root/'binary';binary.write_bytes(b'SYNTHETIC INTERPRETER IDENTITY; NEVER EXECUTED');alias=f.root/'python';alias.symlink_to(binary)
                actual=f.manifest['slots'][3];actual['intent']['python']=dict(path=str(binary),sha256=sha(binary))
                capref=f.put('capacity.json',{})
                request.update(policy=actual['intent']['policy'],policy_id=actual['intent']['policy']['policy_id'],scope='own_cpu32',bank=None,source_root=actual['intent']['output_root'],private_root=actual['private_root'],control_root=str(f.cpu/'admission-staged-integration002/autonomous005'),remote_archive_root=str(m['store'].BACKUP_ROOT),remote_capacity_contract_reference=capref,capacity_contract_reference=capref,remote_python=str(alias))
                f.manifest['preservation_contract']=dict(schema='z1_vega_resident_preservation_contract_v1',mode=a.MODE,backend_package=request['remote_helper_package_reference'],capacity_contract=capref,backup_root=str(m['store'].BACKUP_ROOT),control_root=request['control_root']);f.bind()
                release=f.cpu/'admission-staged-integration002/remaining/release.json';release.parent.mkdir(parents=True);release.write_bytes(f.release.read_bytes());release_ref=dict(path=str(release),sha256=sha(release));request.update(cpu_supervisor_release=release_ref,remote_supervisor_release=release_ref,cpu_supervisor_manifest=f.manifest_ref,reservation_reference=f.put('reservation.json',dict(index=3)))
                with mock.patch.multiple(a,CPU_ROOT=str(f.cpu),SUPERVISOR_SHA=sha(f.consumer),REMOTE_PYTHON=str(alias),RESOLVED_PYTHON=str(binary),PYTHON_SHA=sha(binary)):
                    self.assertEqual(a.remote_authority(request,time.monotonic()+10),f.manifest)
                    bad=dict(request,reservation_reference=f.put('bad-reservation.json',dict(index=6)))
                    with self.assertRaisesRegex(ValueError,'index'):a.remote_authority(bad,time.monotonic()+10)
                    binary.write_bytes(b'CHANGED')
                    with self.assertRaisesRegex(ValueError,'interpreter SHA'):a.remote_authority(request,time.monotonic()+10)
