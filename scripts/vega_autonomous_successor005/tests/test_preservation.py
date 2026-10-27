"""Pure boundaries for the autonomous005 Linux preservation profile."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]

def module(name):
    path = ROOT / (name + '.py')
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

class ReservationTests(unittest.TestCase):
    def setUp(self):
        self.r = module('reservation')
        self.assertIsNotNone(self.r, 'new reservation implementation required')
    def ledger(self):
        r = self.r
        descriptor=r.preissue_descriptor(index=3,release=dict(path=r.CPU_ROOT+'/admission-staged-integration002/remaining/release.json',sha256='b'*64),nonce='a'*32)
        return r.zero_ledger(preissue=descriptor,layout_sha256='c'*64)
    def reservation(self):
        ledger=self.ledger()
        return self.r.make_reservation(ledger, index=3, release={'path':self.r.CPU_ROOT+'/admission-staged-integration002/remaining/release.json','sha256':'b'*64}, nonce='a'*32, ledger_sha256=self.r.digest(ledger), active=[])
    def test_exact_free_equals_R_raw_then_archive_is_one_transaction(self):
        reservation=self.reservation();r=self.r;R=20887633920
        self.assertEqual(reservation['R'],R);self.assertEqual(reservation['G'],461373440)
        for raw,archive,provenance in [(0,0,0),(160*1024**2,0,0),(160*1024**2,192*1024**2,88*1024**2)]:
            usage=dict(raw=raw,archive=archive,provenance=provenance)
            result=r.check_reservation(reservation,usage,free_bytes=R-sum(usage.values()))
            self.assertEqual(result['required_free_bytes'],R-sum(usage.values()))
        with self.assertRaises(ValueError):r.check_reservation(reservation,dict(raw=1,archive=0,provenance=0),free_bytes=R-2)
    def test_overlap_foreign_slot_cap_and_ledger_rebinding_rejected(self):
        r=self.r;reservation=self.reservation();ledger=self.ledger()
        with self.assertRaises(ValueError):r.make_reservation(ledger,index=6,release=reservation['release'],nonce='b'*32,ledger_sha256=r.digest(ledger),active=[reservation])
        for key,value in [('G',1),('index',6),('R',1),('ledger_sha256','e'*64)]:
            bad=copy.deepcopy(reservation);bad[key]=value
            with self.assertRaises(ValueError):r.validate_reservation(bad,ledger)
        for usage in [dict(raw=r.CAPS['raw']+1,archive=0,provenance=0),dict(raw=-1,archive=0,provenance=0),dict(raw=True,archive=0,provenance=0)]:
            with self.assertRaises(ValueError):r.check_reservation(reservation,usage,free_bytes=10**12)
    def test_explicit005_preissue_ledger_schema(self):
        self.assertTrue(callable(getattr(self.r,'preissue_descriptor',None)),'explicit preissue descriptor required')
    def test_actual_cpu_index_three_is_extended_seed_one(self):
        self.assertEqual(self.reservation()['slot']['policy_id'],'extended200-seed2026102301')
    def test_unproved_credit_and_foreign_archive_path_rejected(self):
        r=self.r;ledger=self.ledger();item=dict(path='/tmp/arbitrary',bytes=1,sha256='d'*64,device=1,inode=1,link_count=1,proof=dict(path='/tmp/proof',sha256='e'*64))
        ledger['slots'][3]['retained']['archive']=[item]
        with self.assertRaises(ValueError):r.make_reservation(ledger,index=6,release=dict(path=r.CPU_ROOT+'/admission-staged-integration002/remaining/release.json',sha256='b'*64),nonce='a'*32,ledger_sha256=r.digest(ledger),active=[])

class CurrentUsageTests(unittest.TestCase):
    def test_closed_current_usage_counts_raw_archive_logs_and_rejects_links(self):
        from fixture_support import generic,load
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root,request,ref=generic(Path(tmp).resolve());r=load(root/'reservation.py','usage_reservation')
            ledger=r.zero_ledger(preissue=r.preissue_descriptor(index=3,release=dict(path=r.CPU_ROOT+'/admission-staged-integration002/remaining/release.json',sha256='b'*64),nonce='a'*32),layout_sha256='c'*64)
            record=r.make_reservation(ledger,index=3,release=dict(path=r.CPU_ROOT+'/admission-staged-integration002/remaining/release.json',sha256='b'*64),nonce='a'*32,ledger_sha256=r.digest(ledger),active=[])
            for root_key,filename,data in [('raw','raw.bin',b'r'*3),('archive','terminal-policy-full.tar.gz',b'a'*5),('archive','index.json',b'p'*7),('provenance','slurm.out',b'l'*11)]:
                base=record['roots'][root_key][0] if root_key=='raw' else record['roots'][root_key]
                path=Path(base)/filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            self.assertEqual(r.current_usage(record,deadline=time.monotonic()+5),dict(raw=3,archive=5,provenance=18))
            foreign=Path(tmp).resolve()/'foreign';foreign.write_bytes(b'x'*99)
            self.assertEqual(r.current_usage(record,deadline=time.monotonic()+5),dict(raw=3,archive=5,provenance=18))
            link=Path(record['roots']['provenance'])/'link';link.symlink_to(foreign)
            with self.assertRaises(ValueError):r.current_usage(record,deadline=time.monotonic()+5)
            link.unlink();os.link(foreign,link)
            with self.assertRaises(ValueError):r.current_usage(record,deadline=time.monotonic()+5)
            with self.assertRaises(ValueError):r.current_usage(record,deadline=time.monotonic()+5,max_nodes=1)

class ProfileTests(unittest.TestCase):
    def test_closed_linux_profile_and_all_sources_bound(self):
        entry=module('independent_verify')
        self.assertIsNotNone(entry,'new Linux verifier entry required')
        manifest=json.loads((ROOT/'source-manifest.json').read_bytes())
        self.assertEqual(manifest['entrypoint'],'independent_verify.py')
        actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.py') if 'tests' not in p.relative_to(ROOT).parts}
        self.assertEqual(set(manifest['files']),actual)
        self.assertEqual(entry.MODE,'VEGA_RESIDENT_FULL_ARCHIVE_VEGA_PROCESS_V1')
        for path,info in manifest['files'].items():
            body=(ROOT/path).read_bytes();self.assertEqual(info,dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest()))

if __name__=='__main__':unittest.main()

class ChildBoundaryTests(unittest.TestCase):
    def test_real_child_complete_eof_and_exit_required(self):
        path=ROOT/'archive/transport.py';spec=importlib.util.spec_from_file_location('new_transport',path);transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
        for code in ["import sys;sys.stdout.write('{');sys.exit(0)","import sys;sys.stdout.write('{}');sys.exit(1)"]:
            with self.assertRaises((ValueError,json.JSONDecodeError)):
                result=transport.run_process([sys.executable,'-B','-I','-S','-c',code],b'',deadline=time.monotonic()+5,stdout_cap=1024)
                json.loads(result['stdout'])
        good=transport.run_process([sys.executable,'-B','-I','-S','-c',"print('{}')"],b'',deadline=time.monotonic()+5,stdout_cap=1024)
        self.assertEqual(good['returncode'],0);self.assertNotEqual(good['pid'],__import__('os').getpid())
    def test_generic_preservation_is_implemented_without_cpu_release(self):
        value=module('control_preservation')
        self.assertIsNotNone(value,'independent generic control preservation required')
        self.assertTrue(callable(getattr(value,'preserve_verified',None)))
