"""Full real store/stream/application/publication chain in disposable roots."""
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
try: archive=importlib.import_module('vega_policy_archive.resident_archive')
except ModuleNotFoundError: archive=None
import test_resident_store as store_tests
from test_capacity import ledger


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(archive,'resident coordinator is not implemented')
        self.fixture=store_tests.ResidentStoreTests('test_complete_archive_preserves_originals');self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.f=self.fixture;self.f.success_records()
        self.request=self.f.request
        records={name:dict(value=json.loads((self.f.raw/name).read_bytes()),sha256=hashlib.sha256((self.f.raw/name).read_bytes()).hexdigest()) for name in ('supervisor-terminal.json','intent.json','worker-result.json')}
        self.request.update(_bound_observation=dict(binding=dict(records=records)),
            shared_source=dict(archive_sha256='s'),backend_mode='VEGA_RESIDENT_FULL_ARCHIVE_MAC_STREAM_V1',
            helper_package_reference=dict(path='/helper',sha256='a'*64),capacity_ledger_reference=dict(path='/ledger',sha256='b'*64))
        self.output=self.f.root/'mac';self.output.mkdir()
        self.guard=archive.dependencies()['capacity'].ControlRoot(self.output,free_reader=lambda:10**12,deadline=self.f.deadline)
        self.transport=LocalTransport(self.f)
    def test_complete_real_stream_then_publish_and_read_back_full_remote_receipt(self):
        result=archive.coordinate(self.request,self.transport,self.guard,deadline=self.f.deadline,recheck=lambda:None)
        self.assertEqual(result['status'],'PASS');ref=result['independent_backup']
        self.assertEqual(set(ref),{'path','sha256'})
        value=json.loads(Path(ref['path']).read_bytes())
        self.assertEqual(value['archive_path'],str(self.f.destination/'terminal-policy-full.tar.gz'))
        self.assertEqual(value['members']['intent.json'],hashlib.sha256((self.f.raw/'intent.json').read_bytes()).hexdigest())
        self.assertIn('private-tree/alias',value['node_metadata']);self.assertIn('policy-tree/empty',value['node_metadata'])
        self.assertEqual(value['private_members']['cache'],hashlib.sha256((self.f.private/'cache').read_bytes()).hexdigest())
        self.assertEqual([p.suffix for p in self.output.iterdir()],['.json'])
        self.assertLess(next(self.output.iterdir()).stat().st_size,1024**2)
        self.assertIn('stream',self.transport.actions);self.assertEqual(self.transport.actions[-1],'readcontrol')
    def test_failure_or_forged_identity_never_publishes_local_pass(self):
        for fault in ('archive_sha','job','private','primary','partial','readback','source','application'):
            with self.subTest(fault=fault):
                # Each attempt uses its own exclusive namespace.
                self.f.request['job_id']=str(12345+len(self.transport.actions))
                for name in ('supervisor-terminal.json','intent.json'):
                    value=json.loads((self.f.raw/name).read_bytes())
                    if name=='supervisor-terminal.json':value['allocation']['SLURM_JOB_ID']=self.f.request['job_id']
                    (self.f.raw/name).write_text(json.dumps(value))
                self.f.request['scheduler_identity_capture']=dict(job_id=self.f.request['job_id'],accounting_job_id=self.f.request['job_id'],capture=dict(argv=['scontrol','show','job',self.f.request['job_id'],'-o'],returncode=0,stdout='JobId='+self.f.request['job_id']+' JobState=COMPLETED',stderr=''))
                self.f.destination=self.f.backup/self.request['policy_id']/self.f.raw.name/('job-'+self.request['job_id'])
                records=self.request['_bound_observation']['binding']['records']
                for name in records:records[name]=dict(value=json.loads((self.f.raw/name).read_bytes()),sha256=hashlib.sha256((self.f.raw/name).read_bytes()).hexdigest())
                self.transport.fault=fault
                with self.assertRaises((ValueError,RuntimeError)):archive.coordinate(self.request,self.transport,self.guard,deadline=self.f.deadline,recheck=lambda:None)
                self.assertFalse(any(p.name.endswith('receipt.json') for p in self.output.iterdir()))
    def test_changed_local_authority_after_stream_withholds_publication(self):
        def changed():raise ValueError('authority drift')
        with self.assertRaisesRegex(ValueError,'drift'):archive.coordinate(self.request,self.transport,self.guard,deadline=self.f.deadline,recheck=changed)
        self.assertNotIn('publish',self.transport.actions)
    def test_remote_deadline_uses_own_clock_origin(self):
        with mock.patch.object(archive.time,'monotonic',return_value=900000.):
            self.assertEqual(archive.remote_deadline(7),900007.)
        for value in (0,True,561,-2):
            with self.assertRaises(ValueError):archive.remote_deadline(value)


class LocalTransport:
    """JSON wire round trips and actual reviewed store; only roots/job are synthetic."""
    def __init__(self,fixture):self.f=fixture;self.actions=[];self.fault=None
    def call(self,action,request,*,deadline,consumer=None,**payload):
        self.actions.append(action);request=json.loads(json.dumps(request));payload=json.loads(json.dumps(payload))
        store=archive.dependencies()['store'];destination=store.validate_destination(request)
        if action=='create':
            result=store.create_archive(request,destination,inventory_reader=self.f.inventory,free_reader=lambda:10**12,deadline=deadline)
            if self.fault=='archive_sha':result['archive_sha256']='0'*64
            return result
        if action=='readcontrol':
            ref=payload['reference'];body=Path(ref['path']).read_bytes()
            if self.fault=='readback' and Path(ref['path']).name=='consumer-receipt.json':raise RuntimeError('readback failed')
            value=json.loads(body)
            if self.fault=='job' and Path(ref['path']).name=='before.json':value['job']['job_id']='foreign'
            if self.fault=='private' and Path(ref['path']).name=='before.json':value['private_root']='/foreign'
            if self.fault=='primary' and Path(ref['path']).name=='verification.json':value['archive_sha256']='f'*64
            return dict(body_b64=__import__('base64').b64encode(archive.json_bytes(value) if value!=json.loads(body) else body).decode())
        if action=='stream':
            # Store's sink emits into a synchronous nonseek pipe substitute. Real
            # owned-process backpressure/deadline coverage lives in transport.
            sink=io.BytesIO();store.stream_back(request,destination,sink,deadline=deadline,**payload)
            data=sink.getvalue()
            class Nonseek:
                def __init__(self):self.offset=0
                def read(self,size):result=data[self.offset:self.offset+size];self.offset+=len(result);return result
            result=consumer(Nonseek())
            if self.fault=='application':result['application_records']['policy-tree/intent.json']['policy']={}
            return result
        if action=='inspect':
            value=store.inspect_source(request,inventory_reader=self.f.inventory,deadline=deadline)
            if self.fault=='source':value['nodes']['private-tree/cache']['sha256']='0'*64
            return value
        if action=='publish':
            if self.fault=='partial':raise RuntimeError('partial publication')
            body=__import__('base64').b64decode(payload['body_b64']);path=destination/payload['name']
            result=store.publish_control(path,body,hashlib.sha256(body).hexdigest(),free_reader=lambda:10**12,deadline=deadline)
            return dict(result,path=str(path))
        raise AssertionError(action)

if __name__=='__main__':unittest.main()

class BootstrapAndAuthorityTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(archive)
    def test_pinned_graph_works_isolated_and_source_drift_rejected(self):
        self.assertTrue(callable(getattr(archive,'run',None)),'public run coordinator missing')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()/'helper';root.mkdir()
            source=Path(archive.__file__).parent
            for p in source.rglob('*.py'):
                if 'tests' in p.relative_to(source).parts:continue
                q=root/p.relative_to(source);q.parent.mkdir(exist_ok=True);q.write_bytes(p.read_bytes())
            files={p.relative_to(root).as_posix():dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in root.rglob('*.py')}
            body=archive.json_bytes(dict(schema='z1_vega_policy_archive_package_v1',entrypoint='resident_archive.py',files=files))
            manifest=root/'package-manifest.json';manifest.write_bytes(body)
            ref=dict(path=str(manifest),sha256=hashlib.sha256(body).hexdigest())
            program=archive.BOOTSTRAP+'\nm=load_verified_package('+repr(ref)+')["vega_policy_archive.resident_archive"]\nassert len(m.dependencies())==7\nprint(m.MODE)'
            result=subprocess.run([sys.executable,'-B','-I','-S','-c',program],capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr.decode());self.assertEqual(result.stdout.strip(),archive.MODE.encode())
            for kind in ('source','extra','pyc','symlink'):
                path=root/('transport.py' if kind=='source' else 'foreign.'+('pyc' if kind=='pyc' else 'py'))
                original=path.read_bytes() if path.exists() else None
                if kind=='symlink':path.symlink_to('/never/read')
                else:path.write_bytes(b'raise RuntimeError("must not execute")')
                result=subprocess.run([sys.executable,'-B','-I','-S','-c',program],capture_output=True,timeout=10)
                self.assertNotEqual(result.returncode,0);self.assertNotIn(b'must not execute',result.stderr)
                if original is None:path.unlink()
                else:path.write_bytes(original)
    def test_closed_remote_actions_and_all_current_growth_enforced(self):
        self.assertTrue(callable(getattr(archive,'remote_dispatch',None)),'closed remote dispatcher missing')
        with self.assertRaises(ValueError):archive.remote_dispatch(dict(action='arbitrary',request={},remote_wall_seconds=10),deadline=time.monotonic()+10)

FROZEN=Path(__file__).resolve().parents[3]/'.superpowers/sdd/2026-10-23-z1-overnight-execution/evaluation-staged-integration-repair-proposal002'

def source_module(path,name,location=None):
    value=types.ModuleType(name);value.__file__=str(location or path)
    exec(compile(path.read_bytes(),str(path),'exec'),value.__dict__)
    return value

class UnchangedConsumerTests(PipelineTests):
    # Use the pipeline setup, but do not duplicate its base tests in discovery.
    def _adopt(self,raw,policy):
        self.f.raw=raw;self.request.update(source_root=str(raw),policy=policy,policy_id=policy['policy_id'])
        self.f.destination=self.f.backup/policy['policy_id']/raw.name/('job-'+self.request['job_id'])
        self.f.destination.parent.mkdir(parents=True,exist_ok=True)
        for name in ('supervisor-terminal.json','intent.json','worker-result.json'):
            value=json.loads((raw/name).read_bytes()) if (raw/name).exists() else dict(policy=policy,scope='own_cpu32')
            if name=='supervisor-terminal.json':value['allocation']={'SLURM_JOB_ID':self.request['job_id']}
            if name=='intent.json':value.update(output_root=str(raw),supervisor_context={'release_sha256':'abc'})
            if name=='worker-result.json':value['status']='RAW_REVIEW_REQUIRED'
            (raw/name).write_text(json.dumps(value,sort_keys=True))
        self.request['_bound_observation']['binding']['records']={name:dict(value=json.loads((raw/name).read_bytes()),sha256=hashlib.sha256((raw/name).read_bytes()).hexdigest()) for name in ('supervisor-terminal.json','intent.json','worker-result.json')}
    def test_real_first_seal_and_remaining_gate(self):
        phase=source_module(FROZEN/'tests/test_phase_controls.py','resident_phase_fixture')
        with phase.Fixture('remaining_own32',phase.REMAINING) as f:
            with self.assertRaises(ValueError):f.guard.require_selected(f.guard.load_release(f.release),3)
            raw=f.qualify_first();self._adopt(raw,f.manifest['slots'][0]['intent']['policy'])
            f.refresh_raw_proof()
            result=archive.coordinate(self.request,self.transport,self.guard,deadline=self.f.deadline,recheck=lambda:None)
            f.seal['independent_backup']=result['independent_backup'];f.publish_seal()
            f.guard.verify_qualification(f.manifest['slots'][0]['intent'],f.authority['first_qualification_seal'])
            loaded=f.guard.load_release(f.release)
            for index in phase.REMAINING:f.guard.require_selected(loaded,index)
            reference=f.seal['independent_backup'];path=Path(reference['path']);saved=path.read_bytes()
            for fault in ('status','readback','policy','members','tree','bound_bytes','compact'):
                value=json.loads(saved)
                if fault=='status':value['status']='FAILED'
                if fault=='readback':value['independent_readback']=False
                if fault=='policy':value['policy']={}
                if fault=='members':value['members']={}
                if fault=='tree':value['tree_sha256']='0'*64
                if fault=='compact':value=json.loads(next(self.output.glob('*-receipt.json')).read_bytes())
                path.write_bytes(archive.json_bytes(value)+ (b' ' if fault=='bound_bytes' else b''))
                f.seal['independent_backup']=dict(path=str(path),sha256=reference['sha256'] if fault=='bound_bytes' else hashlib.sha256(path.read_bytes()).hexdigest());f.publish_seal()
                with self.subTest(fault=fault),self.assertRaises(ValueError):f.guard.require_selected(f.guard.load_release(f.release),3)
            path.write_bytes(saved)
    def test_real_32_by_881_npz_audit_and_sealer(self):
        captured=FROZEN/'historical-fixtures/evaluation-preparation/scanner-repair-v001/seal-tools/test_seal_pure.py'
        factory=source_module(captured,'resident_npz_fixture',FROZEN/'seal-tools/test_seal_pure.py')
        with tempfile.TemporaryDirectory() as tmp:
            request=factory.fixture(Path(tmp).resolve(),miss=True)
            raw=Path(request['raw_root']);intent=json.loads((raw/'intent.json').read_bytes());self._adopt(raw,intent['policy'])
            request['intent']['sha256']=factory.w.sha(raw/'intent.json')
            tree=factory.w.tree_digest(factory.w.inventory(raw))
            for role in ('raw','runtime','parity'):
                ref=request[role+'_review'];path=Path(ref['path']);value=json.loads(path.read_bytes())
                value.update(raw_tree_sha256=tree,intent_sha256=request['intent']['sha256']);path.write_bytes(archive.json_bytes(value));ref['sha256']=factory.w.sha(path)
            result=archive.coordinate(self.request,self.transport,self.guard,deadline=self.f.deadline,recheck=lambda:None)
            request['independent_backup']=result['independent_backup']
            seal_ref=factory.w.build(request)
            seal=json.loads(Path(seal_ref['path']).read_bytes())
            self.assertEqual(seal['status'],'QUALIFIED');self.assertEqual(seal['denominators']['contact_801'],31)
            self.assertEqual(seal['denominators']['genuine_no_contact'],1)
            self.assertTrue(seal['unsafe_allowed_for_technical_qualification'])
            factory.w.load_consumer().verify_qualification(json.loads((raw/'intent.json').read_bytes()),{k:seal_ref[k] for k in ('path','sha256')})

class RequestBindingTests(unittest.TestCase):
    def fixture(self,root):
        import ast
        deps=archive.dependencies();path=FROZEN/'historical-fixtures/capacity-split/test_capacity_split.py'
        tree=ast.parse(path.read_text());function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='fixture')
        text=ast.unparse(function).replace("TASK / 'evaluation-argv-mapping-repair-proposal001/cpu-archive-layout-proposal.json'", "FROZEN / 'cpu-archive-layout-proposal.json'")
        space=dict(json=json,c=deps['contract'],r=deps['remote'],FROZEN=FROZEN,REMOTE=11274289152,MAC=3435973836)
        exec(compile(text,str(path),'exec'),space)
        request,manifest,bound=space['fixture'](root)
        cap=deps['capacity']
        contract=dict(schema='z1_vega_resident_capacity_contract_v1',rule_version=cap.RULE,slots=cap.layout_rows(),slot_caps=cap.CAPS,shared_caps=cap.SHARED,immutable_reserve_bytes=cap.IMMUTABLE,mac_envelope_bytes=cap.MAC_ENVELOPE,mac_retained_bytes=cap.MAC_RETAINED,mac_working_bytes=cap.MAC_WORKING)
        package=Path(archive.__file__).parent/'package-manifest.json'
        package_ref=dict(path=str(package),sha256=hashlib.sha256(package.read_bytes()).hexdigest())
        request.update(schema=archive.SCHEMA,backend_mode=archive.MODE,remote_python=archive.REMOTE_PYTHON,max_source_bytes=160*1024**2,max_members=8192,
            mac_control_root=str(root),remote_archive_root=str(deps['store'].BACKUP_ROOT),helper_package_reference=package_ref,
            remote_helper_package_reference=dict(path='/cephhome/eunikhilr/helper/package-manifest.json',sha256=package_ref['sha256']),
            capacity_contract_reference=bound('capacity.json',contract),operation_growth_bytes=280*1024**2)
        request['remote_capacity_contract_reference']=dict(path='/cephhome/eunikhilr/helper/capacity.json',sha256=request['capacity_contract_reference']['sha256'])
        manifest['preservation_contract']=dict(schema='z1_vega_resident_preservation_contract_v1',mode=archive.MODE,backend_package=request['remote_helper_package_reference'],capacity_contract=request['remote_capacity_contract_reference'],backup_root=request['remote_archive_root'],mac_control_root=str(root))
        request['cpu_supervisor_manifest']=bound('manifest.json',manifest)
        request['cpu_supervisor_release']=bound('release.json',dict(status='ADMITTED_ONCE',manifest=request['cpu_supervisor_manifest']))
        request['remote_supervisor_release']=dict(path=archive.CPU_ROOT+'/admission-staged-integration002/first/release.json',sha256=request['cpu_supervisor_release']['sha256'])
        request['remote_archive_destination']=str(deps['store'].validate_destination(request))
        value=ledger();value.update(request_identity_sha256=archive.request_identity(request),release_sha256=request['cpu_supervisor_release']['sha256'],layout_sha256=request['cpu_layout_reference']['sha256'])
        request['capacity_ledger_reference']=bound('ledger.json',value)
        return request,bound
    def test_real_legacy_validator_then_exact_resident_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();request,_=self.fixture(root)
            with mock.patch.object(archive.sys,'platform','darwin'):
                result=archive.validate(request,root/'request.json',time.monotonic()+10)
            self.assertEqual(result['_request_identity_sha256'],archive.request_identity(request))
    def test_owning_authority_ledger_helper_request_and_observation_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();original,_=self.fixture(root)
            cases={'helper':('helper_package_reference','sha256'),'ledger':('capacity_ledger_reference','sha256'),
                   'release':('cpu_supervisor_release','sha256'),'observation':('root_terminal_evidence','sha256'),
                   'scheduler':('scheduler_identity_reference','sha256'),'shared':('shared_source','mac_readback')}
            for fault,(key,field) in cases.items():
                request=copy.deepcopy(original)
                if fault=='shared':request[key][field]['sha256']='0'*64
                else:request[key][field]='0'*64
                with self.subTest(fault=fault),mock.patch.object(archive.sys,'platform','darwin'),self.assertRaises(ValueError):archive.validate(request,root/'request.json',time.monotonic()+10)
            for key,value in [('remote_archive_destination','/foreign'),('remote_python','/foreign'),('backend_mode','fake'),('remote_supervisor_release',dict(path=archive.CPU_ROOT+'/admission-staged-integration002/arbitrary/release.json',sha256=original['cpu_supervisor_release']['sha256'])),('_bound_observation',{})]:
                request=copy.deepcopy(original);request[key]=value
                with self.subTest(key=key),mock.patch.object(archive.sys,'platform','darwin'),self.assertRaises(ValueError):archive.validate(request,root/'request.json',time.monotonic()+10)

class RemoteAuthorityTests(unittest.TestCase):
    """Actual physical supervisor pure gates; only fixed roots/binary are translated."""
    def setUp(self):
        phase=source_module(FROZEN/'tests/test_phase_controls.py','resident_actual_authority')
        self.phase=phase.Fixture();self.addCleanup(self.phase.close);f=self.phase
        self.store_fixture=store_tests.ResidentStoreTests('test_complete_archive_preserves_originals');self.store_fixture.setUp();self.addCleanup(self.store_fixture.doCleanups)
        sf=self.store_fixture;deps=archive.dependencies();cap=deps['capacity'];self.deps=deps
        self.root=f.root;self.output=self.root/'mac';self.output.mkdir()
        self.binary=self.root/'binary';self.binary.write_bytes(b'SYNTHETIC INTERPRETER IDENTITY; NEVER EXECUTED')
        alias=self.root/'python';alias.symlink_to(self.binary)
        for key,value in [('CPU_ROOT',str(f.cpu)),('SUPERVISOR_SHA',phase.sha(f.consumer)),('REMOTE_PYTHON',str(alias)),('RESOLVED_PYTHON',str(self.binary)),('PYTHON_SHA',phase.sha(self.binary))]:
            patch=mock.patch.object(archive,key,value);patch.start();self.addCleanup(patch.stop)
        patch=mock.patch.object(cap,'CPU_ROOT',str(f.cpu));patch.start();self.addCleanup(patch.stop)
        backup=self.root/'backup';backup.mkdir();patch=mock.patch.object(deps['store'],'BACKUP_ROOT',backup);patch.start();self.addCleanup(patch.stop)
        policy=f.manifest['slots'][0]['intent']['policy'];raw=Path(f.manifest['slots'][0]['intent']['output_root']);private=Path(f.manifest['slots'][0]['private_root'])
        raw.mkdir(parents=True);private.mkdir(parents=True);(private/'cache').write_bytes(b'private');(private/'empty').mkdir();(private/'link').symlink_to('/never/follow')
        sf.raw=raw;sf.private=private
        request=dict(sf.request,policy=policy,policy_id=policy['policy_id'],source_root=str(raw),private_root=str(private),schema=archive.SCHEMA,backend_mode=archive.MODE,
            ssh_host='vega',quota_root='/cephhome/eunikhilr',remote_python=str(alias),remote_archive_root=str(backup),mac_control_root=str(self.output),operation_growth_bytes=280*1024**2,
            helper_package_reference=dict(path='/synthetic/local-helper.json',sha256='a'*64),remote_helper_package_reference=dict(path='/synthetic/remote-helper.json',sha256='a'*64),remote_future_reserve_bytes=11274289152)
        contract=dict(schema='z1_vega_resident_capacity_contract_v1',rule_version=cap.RULE,slots=cap.layout_rows(),slot_caps=cap.CAPS,shared_caps=cap.SHARED,immutable_reserve_bytes=cap.IMMUTABLE,mac_envelope_bytes=cap.MAC_ENVELOPE,mac_retained_bytes=cap.MAC_RETAINED,mac_working_bytes=cap.MAC_WORKING)
        request['capacity_contract_reference']=f.put('capacity.json',contract);request['remote_capacity_contract_reference']=request['capacity_contract_reference']
        f.manifest['preservation_contract']=dict(schema='z1_vega_resident_preservation_contract_v1',mode=archive.MODE,backend_package=request['remote_helper_package_reference'],capacity_contract=request['remote_capacity_contract_reference'],backup_root=str(backup),mac_control_root=str(self.output))
        f.manifest['slots'][0]['intent']['python']=dict(path=str(self.binary),sha256=phase.sha(self.binary));f.bind()
        release=f.cpu/'admission-staged-integration002/first/release.json';release.parent.mkdir(parents=True);release.write_bytes(f.release.read_bytes())
        request.update(cpu_supervisor_manifest=f.manifest_ref,cpu_supervisor_release=dict(path=str(release),sha256=phase.sha(release)),remote_supervisor_release=dict(path=str(release),sha256=phase.sha(release)))
        layout=dict(schema='z1_horizon_cpu_archive_layout_v1',root=str(f.cpu),slots=cap.layout_rows());request.update(cpu_layout=layout,cpu_layout_reference=f.put('layout.json',layout))
        sf.request=request;sf.success_records()
        (raw/'intent.json').write_bytes(archive.json_bytes(dict(policy=policy,scope='own_cpu32',output_root=str(raw),supervisor_context={'release_sha256':request['cpu_supervisor_release']['sha256']})))
        request['_bound_observation']=dict(binding=dict(records={name:dict(sha256=phase.sha(raw/name),value=json.loads((raw/name).read_bytes())) for name in ('supervisor-terminal.json','intent.json','worker-result.json')}))
        request['shared_source']=dict(archive_sha256='fixture-shared-proof-already-validated-locally')
        request['remote_archive_destination']=str(deps['store'].validate_destination(request));sf.destination=Path(request['remote_archive_destination'])
        value=ledger()
        for row,actual in zip(value['slots'],cap.layout_rows()):row.update(actual)
        value.update(request_identity_sha256=archive.request_identity(request),release_sha256=request['cpu_supervisor_release']['sha256'],layout_sha256=request['cpu_layout_reference']['sha256'])
        request['capacity_ledger_reference']=f.put('ledger.json',value);request['_ledger']=value;request['_ledger_body_b64']=__import__('base64').b64encode(Path(request['capacity_ledger_reference']['path']).read_bytes()).decode()
        self.request=request;self.sf=sf;self.deadline=time.monotonic()+30
        patch=mock.patch.object(deps['remote'],'inventory',side_effect=sf.inventory);patch.start();self.addCleanup(patch.stop)
        self.initial=cap.remaining_budget(value)+1024**2
        patch=mock.patch.object(archive,'quota_headroom',side_effect=lambda:self.initial-archive._allocation_usage(sf.destination,self.deadline)['total']);patch.start();self.addCleanup(patch.stop)
    def test_actual_supervisor_load_release_require_selected_and_interpreter_alias(self):
        actual=archive.remote_authority(self.request,self.deadline)
        self.assertEqual(actual,self.phase.manifest)
        self.binary.write_bytes(b'changed binary')
        with self.assertRaisesRegex(ValueError,'interpreter SHA'):archive.remote_authority(self.request,self.deadline)
    def test_wrong_phase_release_and_actual_authority_drift_reject(self):
        request=copy.deepcopy(self.request);path=self.phase.cpu/'admission-staged-integration002/remaining/release.json';path.parent.mkdir(parents=True);path.write_bytes(Path(request['remote_supervisor_release']['path']).read_bytes())
        request['remote_supervisor_release']['path']=str(path)
        with self.assertRaisesRegex(ValueError,'phase'):archive.remote_authority(request,self.deadline)
        self.phase.consumer.write_bytes(b'changed supervisor')
        with self.assertRaisesRegex(ValueError,'supervisor source SHA'):archive.remote_authority(self.request,self.deadline)
    def complete(self):
        calls=[]
        class Adapter:
            def call(inner,action,request,*,deadline,consumer=None,**payload):
                calls.append(action);packet=json.loads(json.dumps(dict(action=action,request=request,payload=payload,credit_proofs={})))
                if action=='stream':
                    sink=io.BytesIO();archive.remote_dispatch(packet,deadline=deadline,sink=sink)
                    return consumer(io.BytesIO(sink.getvalue()))
                return json.loads(archive.json_bytes(archive.remote_dispatch(packet,deadline=deadline)))
        guard=self.deps['capacity'].ControlRoot(self.output,free_reader=lambda:10**12,deadline=self.deadline)
        result=archive.coordinate(self.request,Adapter(),guard,deadline=self.deadline,recheck=lambda:None)
        self.assertEqual(result['status'],'PASS');self.assertEqual(calls.count('create'),1)
        full=json.loads(Path(result['independent_backup']['path']).read_bytes());self.assertEqual(full['archive_sha256'],result['archive_sha256'])
        return result
    def test_full_real_serialized_remote_chain_including_capacity_and_actual_release(self):
        self.complete()
    def test_expired_and_low_ceph_or_growth_deny_without_complete_archive(self):
        with self.assertRaises(TimeoutError):archive.remote_dispatch(dict(action='create',request=self.request),deadline=time.monotonic()-1)
        self.initial=self.deps['capacity'].remaining_budget(self.request['_ledger'])-1
        with self.assertRaisesRegex(ValueError,'capacity preflight'):archive.remote_dispatch(dict(action='create',request=self.request),deadline=self.deadline)
        self.assertFalse(self.sf.destination.exists())
        self.initial=10**12;request=copy.deepcopy(self.request);request['operation_growth_bytes']=1
        value=request['_ledger'];value['request_identity_sha256']=archive.request_identity(request);body=archive.json_bytes(value);request['capacity_ledger_reference']['sha256']=hashlib.sha256(body).hexdigest();request['_ledger_body_b64']=__import__('base64').b64encode(body).decode()
        with self.assertRaises(ValueError):archive.remote_dispatch(dict(action='create',request=request),deadline=self.deadline)
        self.assertFalse((self.sf.destination/'index.json').exists());self.assertLessEqual(archive._allocation_usage(self.sf.destination,self.deadline)['total'],1)

    def test_credits_require_current_source_archive_and_full_proof_closure(self):
        result=self.complete();compact_path=next(self.output.glob('*-receipt.json'));body=compact_path.read_bytes()
        proof=dict(path=str(compact_path),sha256=hashlib.sha256(body).hexdigest())
        path=self.sf.raw/'intent.json';info=path.stat()
        item=dict(path=str(path),bytes=info.st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),device=info.st_dev,inode=info.st_ino,link_count=info.st_nlink,proof=proof)
        request=copy.deepcopy(self.request);value=request['_ledger'];value['slots'][0]['retained']['raw']=[item]
        value['request_identity_sha256']=archive.request_identity(request);raw=archive.json_bytes(value)
        request['capacity_ledger_reference']=self.phase.put('next-ledger.json',value)
        request['_ledger_body_b64']=__import__('base64').b64encode(Path(request['capacity_ledger_reference']['path']).read_bytes()).decode()
        proofs={proof['sha256']:__import__('base64').b64encode(body).decode()}
        validated=archive.remote_validate(request,self.deadline,proofs)
        self.assertEqual(self.deps['capacity'].remaining_budget(validated),20887633920-info.st_size)
        # A self-consistent changed PASS flag, even rebound by the ledger, cannot
        # replace the complete prior proof closure.
        forged=json.loads(body);forged['independent_backup']['sha256']='0'*64;forged_body=archive.json_bytes(forged)
        bad=copy.deepcopy(request);newproof=dict(path=str(compact_path),sha256=hashlib.sha256(forged_body).hexdigest());bad['_ledger']['slots'][0]['retained']['raw'][0]['proof']=newproof
        ledger_body=archive.json_bytes(bad['_ledger']);bad['capacity_ledger_reference']['sha256']=hashlib.sha256(ledger_body).hexdigest();bad['_ledger_body_b64']=__import__('base64').b64encode(ledger_body).decode()
        with self.assertRaises(ValueError):archive.remote_validate(bad,self.deadline,{newproof['sha256']:__import__('base64').b64encode(forged_body).decode()})
        # A missing archive invalidates even source-byte credits.
        archived=Path(result['create']['archive_path']);saved=archived.read_bytes();archived.unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):archive.remote_validate(request,self.deadline,proofs)
        archived.write_bytes(saved)
        info=path.stat();os.chmod(path,0o400)
        with self.assertRaisesRegex(ValueError,'metadata|identity'):archive.remote_validate(request,self.deadline,proofs)
