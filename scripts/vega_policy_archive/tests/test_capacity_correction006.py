"""Disjoint ruling003 admission through the real stdlib archive APIs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vega_policy_archive import capacity, resident_archive, resident_store
import test_capacity
import test_resident_archive


def ledger():
    value = test_capacity.ledger()
    value['rule_version'] = 'capacity-ledger-ruling003'
    return value


def contract():
    rows = [{key: row[key] for key in ('policy_id', 'scope', 'bank', 'run_root', 'private_root')}
            for row in ledger()['slots']]
    return dict(schema='z1_vega_resident_capacity_contract_v1',
                rule_version='capacity-ledger-ruling003', slots=rows,
                slot_caps=dict(raw=167772160, archive=201326592, provenance=92274688),
                shared_caps=dict(derived=671088640, controls=67108864),
                immutable_reserve_bytes=5133828096, mac_envelope_bytes=1400897536,
                mac_retained_bytes=67108864, mac_working_bytes=92274688)


def item(path, size, inode):
    return dict(path=path, bytes=size, sha256='d'*64, device=1, inode=inode,
                link_count=1, proof=dict(path='/proof', sha256='e'*64))


class DisjointCapacityTests(unittest.TestCase):
    def test_current_budget_api_counts_training_and_evaluation_once(self):
        value = ledger(); value['rule_version'] = capacity.RULE
        self.assertEqual(capacity.remaining_budget(value), 14176747520)

    def test_zero_ledger_reserves_disjoint_union_with_all_18_slots(self):
        value = ledger()
        self.assertEqual(capacity.remaining_budget(value), 14176747520)
        for row in value['slots']:
            row['status'] = 'FAILED_NO_RETRY'
        self.assertEqual(capacity.remaining_budget(value), 14176747520)
        value['staging_bytes'] = 731
        self.assertEqual(capacity.remaining_budget(value), 14176748251)
        value['slots'].pop()
        with self.assertRaisesRegex(ValueError, 'eighteen'):
            capacity.remaining_budget(value)

    def test_only_materialized_bytes_release_capacity_never_unused_caps(self):
        value = ledger()
        value['slots'][0].update(status='COMPLETE')
        value['slots'][0]['retained']['raw'] = [item(value['slots'][0]['run_root']+'/a', 31, 1)]
        self.assertEqual(capacity.remaining_budget(value), 14176747489)
        value['slots'][1]['retained']['raw'] = [item(value['slots'][1]['run_root']+'/b', 31, 1)]
        self.assertEqual(capacity.remaining_budget(value), 14176747489)

    def test_old_rule_invalid_credits_booleans_and_cap_releases_reject(self):
        changes = [lambda v: v.update(rule_version='capacity-ledger-ruling002'),
                   lambda v: v.update(staging_bytes=True),
                   lambda v: v.update(staging_bytes=-1),
                   lambda v: v['slots'][0]['caps'].update(raw=0),
                   lambda v: v['slots'][0]['caps'].update(raw=True),
                   lambda v: v['slots'][0]['retained']['raw'].append(dict(bytes=12)),
                   lambda v: v['slots'][0]['retained']['raw'].append(item(v['slots'][0]['run_root']+'/a', True, 1)),
                   lambda v: v['slots'][0]['retained']['raw'].append(item(v['slots'][0]['run_root']+'/a', 167772161, 1))]
        for change in changes:
            value = ledger(); change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                capacity.remaining_budget(value)

    def test_late_phase_guard_keeps_entire_training_allowance_and_floor(self):
        value = ledger(); inode = 0
        for row in value['slots']:
            for role, size in row['caps'].items():
                inode += 1
                path = row['run_root']+'/retained' if role == 'raw' else '/backup/%d'%inode
                row['retained'][role] = [item(path, size, inode)]
        for role, allocation in value['shared'].items():
            inode += 1
            allocation['retained'] = [item('/shared/'+role, allocation['cap'], inode)]
        self.assertEqual(capacity.remaining_budget(value), 5133828096)
        result = capacity.operation_budget(value, 1)
        self.assertEqual(result['protected_bytes'], 5133828096)
        self.assertEqual(result['future_reserve_bytes'], 4563402752)
        for growth in (True, 0, -1, 1199570945):
            with self.subTest(growth=growth), self.assertRaises(ValueError):
                capacity.operation_budget(value, growth)

    def test_exact_versioned_contract_accepts_new_rule_and_rejects_old_rule(self):
        request = dict(cpu_supervisor_release=dict(sha256='b'*64), cpu_layout_reference=dict(sha256='c'*64))
        value = ledger(); value['request_identity_sha256'] = resident_archive.request_identity(request)
        resident_archive.capacity_contract(request, contract(), value)
        old = contract(); old.update(rule_version='capacity-ledger-ruling002', immutable_reserve_bytes=11844714496)
        with self.assertRaisesRegex(ValueError, 'capacity contract'):
            resident_archive.capacity_contract(request, old, value)
        value['rule_version'] = 'capacity-ledger-ruling002'
        with self.assertRaisesRegex(ValueError, 'schema/rule'):
            resident_archive.capacity_contract(request, contract(), value)

    def test_actual_versioned_contract_matches_enforced_allocations(self):
        path = Path(capacity.__file__).parent/'capacity-contract-ruling003.json'
        self.assertTrue(path.is_file(), 'reviewable ruling003 contract must be present')
        self.assertEqual(json.loads(path.read_bytes()), contract())

    def test_operation_reserve_subtracts_growth_and_adds_floor_once(self):
        value = ledger()
        result = capacity.operation_budget(value, 293601280)
        self.assertEqual(result['remaining_bytes'], 14176747520)
        self.assertEqual(result['protected_bytes'], 13883146240)
        self.assertEqual(result['future_reserve_bytes'], 13312720896)

    def request(self):
        return dict(max_source_bytes=167772160, max_archive_bytes=201326592,
                    max_metadata_bytes=4194304, max_evidence_bytes=8388608, max_members=8192,
                    future_reserve_bytes=0, remote_future_reserve_bytes=4563402752)

    def test_store_minimum_keeps_training_allowance_and_floor_once(self):
        self.assertEqual(resident_store._bounds(self.request()), 5133828096)
        self.assertEqual(resident_store._bounds(dict(self.request(), remote_future_reserve_bytes=11274289152)), 11844714496)

    def test_compatibility_minimum_accepts_disjoint_training_reserve(self):
        request = self.request()
        policy = resident_archive.dependencies()['policy']
        self.assertEqual(policy.remote_request_payload(request, {})['future_reserve_bytes'], 4563402752)
        for reserve in (4563402751, True):
            with self.subTest(reserve=reserve), self.assertRaises(ValueError):
                policy.remote_request_payload(dict(request, remote_future_reserve_bytes=reserve), {})

    def test_new_minimum_does_not_override_higher_selected_manifest_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            request,_=test_resident_archive.RequestBindingTests().fixture(root)
            request['remote_future_reserve_bytes']=4563402752
            with mock.patch.object(resident_archive.sys,'platform','darwin'), self.assertRaisesRegex(ValueError,'selected actual manifest'):
                resident_archive.validate(request,root/'request.json',time.monotonic()+10)


class DisjointCapacityIOTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_resident_archive.RemoteAuthorityTests('test_actual_supervisor_load_release_require_selected_and_interpreter_alias')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture; self.request = f.request
        self.request['remote_future_reserve_bytes'] = 4563402752
        value = self.request['_ledger']; value['rule_version'] = 'capacity-ledger-ruling003'
        value['request_identity_sha256'] = resident_archive.request_identity(self.request)
        self.request['capacity_ledger_reference'] = f.phase.put('ruling003-ledger.json', value)
        self.request['_ledger_body_b64'] = __import__('base64').b64encode(Path(self.request['capacity_ledger_reference']['path']).read_bytes()).decode()
        # The fixture translates only the closed CPU root; the public contract
        # must carry that same translated map before actual authority is rebound.
        actual_contract = contract()
        actual_contract['slots'] = capacity.layout_rows()
        self.request['capacity_contract_reference'] = f.phase.put('ruling003-capacity.json', actual_contract)
        self.request['remote_capacity_contract_reference'] = self.request['capacity_contract_reference']
        f.phase.manifest['preservation_contract']['capacity_contract'] = self.request['remote_capacity_contract_reference']
        for slot in f.phase.manifest['slots']:
            slot['limits']['future_reserve_bytes'] = 4563402752
        f.phase.bind()
        release = Path(self.request['remote_supervisor_release']['path'])
        release.write_bytes(f.phase.release.read_bytes())
        self.request['cpu_supervisor_manifest'] = f.phase.manifest_ref
        self.request['cpu_supervisor_release']['sha256'] = hashlib.sha256(release.read_bytes()).hexdigest()
        self.request['remote_supervisor_release']['sha256'] = self.request['cpu_supervisor_release']['sha256']
        # This synthetic fixture writes no scientific payload; bind its changed
        # release in the actual source application record and original ledger.
        path = f.sf.raw/'intent.json'; intent = json.loads(path.read_bytes())
        intent['supervisor_context']['release_sha256'] = self.request['cpu_supervisor_release']['sha256']
        path.write_bytes(resident_archive.json_bytes(intent))
        self.request['_bound_observation']['binding']['records']['intent.json'] = dict(value=intent, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        value['release_sha256'] = self.request['cpu_supervisor_release']['sha256']
        value['request_identity_sha256'] = resident_archive.request_identity(self.request)
        self.request['capacity_ledger_reference'] = f.phase.put('ruling003-ledger.json', value)
        self.request['_ledger_body_b64'] = __import__('base64').b64encode(Path(self.request['capacity_ledger_reference']['path']).read_bytes()).decode()
        f.initial = 14176747520

    def dispatch(self):
        return resident_archive.remote_dispatch(dict(action='create', request=self.request, credit_proofs={}), deadline=self.fixture.deadline)

    def test_exact_disjoint_headroom_admits_and_each_write_decrements_actual_quota(self):
        f = self.fixture; samples = []
        def quota():
            usage = resident_archive._allocation_usage(f.sf.destination, f.deadline)['total']
            samples.append(usage)
            return 14176747520-usage
        with mock.patch.object(resident_archive, 'quota_headroom', side_effect=quota):
            result = self.dispatch()
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertGreater(len(set(samples)), 3)
        self.assertGreater(samples[-1], samples[0])
        self.assertLessEqual(samples[-1], 293601280)

    def test_quota_drop_during_write_preserves_partial_without_complete_index(self):
        f = self.fixture
        def quota():
            usage = resident_archive._allocation_usage(f.sf.destination, f.deadline)['total']
            if (f.sf.destination/'terminal-policy-full.tar.gz').exists():
                return 14176747519-usage
            return 14176747520-usage
        with mock.patch.object(resident_archive, 'quota_headroom', side_effect=quota), self.assertRaisesRegex(ValueError, 'capacity preflight'):
            self.dispatch()
        self.assertTrue((f.sf.destination/'terminal-policy-full.tar.gz').exists())
        self.assertFalse((f.sf.destination/'index.json').exists())

    def test_declared_growth_cap_is_still_enforced(self):
        f = self.fixture; self.request['operation_growth_bytes'] = 1
        value = self.request['_ledger']; value['request_identity_sha256'] = resident_archive.request_identity(self.request)
        self.request['capacity_ledger_reference'] = f.phase.put('small-growth-ledger.json', value)
        self.request['_ledger_body_b64'] = __import__('base64').b64encode(Path(self.request['capacity_ledger_reference']['path']).read_bytes()).decode()
        with self.assertRaises(ValueError): self.dispatch()
        self.assertLessEqual(resident_archive._allocation_usage(f.sf.destination, f.deadline)['total'], 1)
        self.assertFalse((f.sf.destination/'index.json').exists())

    def test_uncovered_staging_rejects_before_any_archive_write(self):
        f = self.fixture; value = self.request['_ledger']; value['staging_bytes'] = 1
        self.request['capacity_ledger_reference'] = f.phase.put('staging-ledger.json', value)
        self.request['_ledger_body_b64'] = __import__('base64').b64encode(Path(self.request['capacity_ledger_reference']['path']).read_bytes()).decode()
        with self.assertRaisesRegex(ValueError, 'capacity preflight'): self.dispatch()
        self.assertFalse(f.sf.destination.exists())


if __name__ == '__main__':
    unittest.main()
