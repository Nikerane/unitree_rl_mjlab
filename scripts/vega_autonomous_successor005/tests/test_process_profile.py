import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from fixture_support import ROOT,body,generic,manifest,sha,write

class ProcessProfileTests(unittest.TestCase):
    def run_fixture(self,root,request,ref):
        return subprocess.run([sys.executable,'-B','-I','-S',str(root/'independent_verify.py'),'--request',ref['path'],'--output',request['control_root']],capture_output=True,timeout=30)
    def test_actual_separate_writer_and_verifier_generic_full_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,request,ref=generic(Path(tmp).resolve());result=self.run_fixture(root,request,ref)
            self.assertEqual(result.returncode,0,result.stderr.decode())
            reference=json.loads(result.stdout);self.assertEqual(set(reference),{'path','sha256'});self.assertEqual(sha(reference['path']),reference['sha256'])
            receipt=json.loads(Path(reference['path']).read_bytes());self.assertEqual(receipt['kind'],'CONTROL_ONLY');self.assertFalse(receipt['technical_qualification'])
            self.assertIn('private-tree/metadata',receipt['node_metadata']);self.assertIn('policy-tree/empty',receipt['node_metadata'])
            provenance=json.loads(Path(receipt['provenance']['path']).read_bytes());writer=provenance['writer_identity'];verifier=provenance['verifier_identity']
            self.assertEqual(writer.get('invocation',{}).get('argv0'),'-c');self.assertEqual(writer['entrypoint']['path'],str(root/'archive/resident_archive.py'));self.assertEqual(writer['platform'],sys.platform);self.assertEqual(verifier['platform'],sys.platform);self.assertEqual(writer['ppid'],verifier['pid']);self.assertNotEqual(writer['pid'],verifier['pid'])
            self.assertEqual(verifier['interpreter']['sha256'],sha(Path(sys.executable).resolve()));self.assertEqual(provenance['stream_transport_exit'],0);self.assertTrue(provenance['complete_eof'])
            self.assertNotIn('cpu_supervisor_release',provenance['request'])
    def test_source_rebound_extra_bytecode_foreign_authority_and_mac_operative_ref(self):
        for fault in ('rebound','extra','bytecode','symlink','authority','mac'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as tmp:
                root,request,ref=generic(Path(tmp).resolve())
                if fault=='rebound':(root/'archive/transport.py').write_bytes(b'raise RuntimeError("MUST_NOT_EXECUTE")')
                if fault=='extra':(root/'extra.py').write_bytes(b'raise RuntimeError("MUST_NOT_EXECUTE")')
                if fault=='bytecode':(root/'bad.pyc').write_bytes(b'bad')
                if fault=='symlink':(root/'bad').symlink_to('/unread')
                if fault=='authority':
                    old=request['preservation_authority'];request['preservation_authority']=write(Path(tmp).resolve()/'foreign-authority.json',json.loads(Path(old['path']).read_bytes()))
                if fault=='mac':request['capacity_ledger_reference']=dict(path='/Users/operative/ledger.json',sha256='f'*64)
                ref=write(ref['path'],request);result=self.run_fixture(root,request,ref)
                self.assertNotEqual(result.returncode,0);self.assertFalse(result.stdout);self.assertNotIn(b'MUST_NOT_EXECUTE',result.stderr)
    def test_process_pid_collision_rejected(self):
        entry=__import__('test_preservation').module('independent_verify')
        # Pure pinned loader authenticates generic authority before any source.
        with tempfile.TemporaryDirectory() as tmp:
            root,request,ref=generic(Path(tmp).resolve());entry=__import__('fixture_support').load(root/'independent_verify.py','translated_entry');loaded=entry.load_verified_package(request['helper_package_reference'],request,str(Path(ref['path']).parent));a=loaded['vega_policy_archive.resident_archive']
            identity=a.process_identity(request)
            with self.assertRaises(ValueError):a.validate_process_pair(identity,identity)
    def test_late_writer_exit_partial_json_and_source_freshness_hold(self):
        for fault in ('late-exit','partial-json','source-drift','missing-private'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as tmp:
                root,request,ref=generic(Path(tmp).resolve());path=root/'control_preservation.py';source=path.read_text()
                if fault=='late-exit':source=source.replace('free_reader();return result',"free_reader();raise SystemExit(1)")
                if fault=='partial-json':source=source.replace("result['writer_identity']=a.process_identity(request,role='writer');body=a.json_bytes(result['writer_identity'])","__import__('sys').stdout.write('{');raise SystemExit(0)\n        result['writer_identity']=a.process_identity(request,role='writer');body=a.json_bytes(result['writer_identity'])")
                if fault=='source-drift':source=source.replace("if action=='inspect':", "if action=='inspect':\n        (Path(request['private_root'])/'metadata').write_bytes(b'CHANGED')")
                if fault=='missing-private':source=source.replace("if action=='inspect':", "if action=='inspect':\n        (Path(request['private_root'])/'metadata').unlink()")
                path.write_text(source);reference=manifest(root);authority=json.loads(Path(request['preservation_authority']['path']).read_bytes());authority['helper_package']=reference;request['preservation_authority']=write(request['preservation_authority']['path'],authority);request['helper_package_reference']=reference;request['remote_helper_package_reference']=reference;ref=write(ref['path'],request)
                result=self.run_fixture(root,request,ref);self.assertNotEqual(result.returncode,0);self.assertFalse(result.stdout)
                self.assertFalse((Path(request['remote_archive_destination'])/'consumer-receipt.json').exists())
    def test_crc_trailer_and_full_private_metadata_gates(self):
        from fixture_support import load
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root,request,ref=generic(Path(tmp).resolve());result=self.run_fixture(root,request,ref);self.assertEqual(result.returncode,0,result.stderr.decode())
            entry=load(root/'independent_verify.py','translated_reader');loaded=entry.load_verified_package(request['helper_package_reference'],request,str(Path(ref['path']).parent));a=loaded['vega_policy_archive.resident_archive'];reader=a.dependencies()['reader']
            destination=Path(request['remote_archive_destination']);archive=(destination/'terminal-policy-full.tar.gz').read_bytes();before=json.loads((destination/'before.json').read_bytes())
            bad_crc=bytearray(archive);bad_crc[-8]^=1
            for data in (bytes(bad_crc),archive+b'foreign trailing bytes',archive[:-1]):
                with self.subTest(size=len(data)),self.assertRaises(ValueError):reader.verify_stream(io.BytesIO(data),before,max_archive_bytes=request['max_archive_bytes'],max_source_bytes=request['max_source_bytes'],max_metadata_bytes=request['max_metadata_bytes'],max_evidence_bytes=request['max_evidence_bytes'],application_names=[],deadline=time.monotonic()+5)
            for fault in ('missing-node','restore-metadata'):
                bad=copy.deepcopy(before)
                if fault=='missing-node':bad['nodes'].pop('private-tree/metadata')
                else:bad['nodes']['private-tree/metadata']['restore_metadata']['xattrs_b64']={'user.extra':'Zm9yZWlnbg=='}
                with self.subTest(fault=fault),self.assertRaises(ValueError):reader.verify_stream(io.BytesIO(archive),bad,max_archive_bytes=request['max_archive_bytes'],max_source_bytes=request['max_source_bytes'],max_metadata_bytes=request['max_metadata_bytes'],max_evidence_bytes=request['max_evidence_bytes'],application_names=[],deadline=time.monotonic()+5)
