"""Pure fixed allocation arithmetic and real exclusive aggregate controls."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    capacity = importlib.import_module('vega_policy_archive.capacity')
except ModuleNotFoundError:
    capacity = None


def ledger():
    rows = []
    for arm in ('native50', 'extended200'):
        for seed in (2026102301, 2026102302, 2026102303):
            policy = '%s-seed%d' % (arm, seed)
            for scope, bank in [('own_cpu32', None), ('science160', 20261041), ('science160', 2026101601)]:
                leaf = 'own_cpu32-staged-integration002' if bank is None else 'science160-bank%d' % bank
                base = '/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002'
                rows.append(dict(policy_id=policy, scope=scope, bank=bank,
                    run_root=base+'/runs/'+policy+'/'+leaf, private_root=base+'/private/'+policy+'/'+leaf,
                    caps=dict(raw=160*1024**2, archive=192*1024**2, provenance=88*1024**2),
                    retained=dict(raw=[], archive=[], provenance=[])))
    return dict(schema='z1_vega_resident_capacity_ledger_v1', rule_version='capacity-ledger-ruling003',
        request_identity_sha256='a'*64, release_sha256='b'*64, layout_sha256='c'*64,
        slots=rows, shared=dict(derived=dict(cap=640*1024**2, retained=[]), controls=dict(cap=64*1024**2, retained=[])),
        staging_bytes=0)


class CapacityTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(capacity, 'capacity module is not implemented')
    def test_all_18_slots_and_unused_caps_stay_reserved(self):
        value=ledger()
        self.assertEqual(capacity.remaining_budget(value), 14176747520)
        self.assertEqual(len(value['slots']),18)
        value['slots'][0]['status']='FAILED'
        self.assertEqual(capacity.remaining_budget(value),14176747520)
    def test_supported_bytes_only_and_global_inode_dedup(self):
        value=ledger(); ref=dict(path='/proof',sha256='d'*64)
        item=dict(path=value['slots'][0]['run_root']+'/a',bytes=31,sha256='e'*64,
                  device=1,inode=2,link_count=1,proof=ref)
        value['slots'][0]['retained']['raw']=[item]
        self.assertEqual(capacity.remaining_budget(value),14176747520-31)
        value['slots'][1]['retained']['raw']=[dict(item,path=value['slots'][1]['run_root']+'/b')]
        self.assertEqual(capacity.remaining_budget(value),14176747520-31)
        value['slots'][1]['retained']['provenance']=[value['slots'][1]['retained']['raw'].pop()]
        with self.assertRaisesRegex(ValueError,'inode'): capacity.remaining_budget(value)
    def test_invalid_slots_caps_and_unsupported_credits(self):
        for change in ('missing','duplicate','cap','negative','boolean','unsupported'):
            value=ledger()
            if change=='missing': value['slots'].pop()
            if change=='duplicate': value['slots'][-1]=copy.deepcopy(value['slots'][0])
            if change=='cap': value['slots'][0]['caps']['raw']-=1
            if change=='negative': value['staging_bytes']=-1
            if change=='boolean': value['staging_bytes']=True
            if change=='unsupported': value['slots'][0]['retained']['raw']=[dict(bytes=5)]
            with self.subTest(change=change),self.assertRaises(ValueError): capacity.remaining_budget(value)
    def test_growth_reserve_does_not_double_add_floor(self):
        result=capacity.operation_budget(ledger(),192*1024**2+88*1024**2)
        self.assertEqual(result['protected_bytes'],14176747520-280*1024**2)
        self.assertEqual(result['future_reserve_bytes']+570425344,result['protected_bytes'])
        with self.assertRaises(ValueError): capacity.operation_budget(ledger(),-1)
    def test_exclusive_controls_and_actual_64_mib_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); guard=capacity.ControlRoot(root,free_reader=lambda:10**12,deadline=time.monotonic()+20)
            self.assertEqual(guard.write('one.json',b'{}')['bytes'],2)
            with self.assertRaises(FileExistsError): guard.write('one.json',b'{}')
            with (root/'existing').open('wb') as stream: stream.truncate(64*1024**2-2)
            with self.assertRaisesRegex(ValueError,'aggregate'): guard.write('two.json',b'x')
            self.assertFalse((root/'two.json').exists())
    def test_reject_links_paths_oversize_deadline_and_low_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); guard=capacity.ControlRoot(root,free_reader=lambda:10**12,deadline=time.monotonic()+20)
            for name in ('../a','/tmp/a','a/b'):
                with self.assertRaises(ValueError): guard.write(name,b'x')
            with self.assertRaises(ValueError): guard.write('big',b'x'*(1024**2+1))
            (root/'link').symlink_to('/tmp')
            with self.assertRaises(ValueError): guard.write('small',b'x')
        with tempfile.TemporaryDirectory() as tmp:
            guard=capacity.ControlRoot(Path(tmp).resolve(),free_reader=lambda:1400897536-1,deadline=time.monotonic()+20)
            with self.assertRaisesRegex(ValueError,'Mac'): guard.write('small',b'x')
            guard.deadline=time.monotonic()-1
            with self.assertRaises(TimeoutError): guard.write('small',b'x')

if __name__=='__main__': unittest.main()

class OperationLeaseTests(unittest.TestCase):
    def test_one_control_root_cannot_reuse_growth_concurrently(self):
        self.assertTrue(callable(getattr(capacity.ControlRoot,'lease',None)),'whole operation must hold the shared allocation lock')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();first=capacity.ControlRoot(root,free_reader=lambda:10**12,deadline=time.monotonic()+10)
            second=capacity.ControlRoot(root,free_reader=lambda:10**12,deadline=time.monotonic()+.05)
            with first.lease():
                self.assertEqual(first.write('receipt',b'{}')['bytes'],2)
                with self.assertRaises(TimeoutError):
                    with second.lease():pass
