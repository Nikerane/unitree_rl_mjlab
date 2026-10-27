"""Synthetic local IO tests; no scheduler, model, or remote execution."""
import copy
import errno
import io
import gzip
import tarfile
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    store = importlib.import_module('vega_policy_archive.resident_store')
except ModuleNotFoundError:
    store = None


class ResidentStoreTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(store, 'exclusive resident store is not implemented')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.raw = self.root / 'runs' / 'native50-seed2026102301' / 'own_cpu32-staged-integration002'
        self.private = self.root / 'private' / 'native50-seed2026102301' / self.raw.name
        self.raw.mkdir(parents=True)
        self.private.mkdir(parents=True)
        (self.raw / 'empty').mkdir()
        (self.private / 'cache').write_bytes(b'partial cache\x00')
        (self.private / 'alias').symlink_to('/unread/target')
        policy = dict(policy_id='native50-seed2026102301', arm='native50', training_seed=2026102301,
                      task='unused in fixture', native_window_s=.05)
        self.request = dict(policy=policy, policy_id=policy['policy_id'], scope='own_cpu32', bank=None,
                            source_root=str(self.raw), private_root=str(self.private), job_id='12345',
                            guard_status='FAILED_NO_RETRY', guard_returncode=1,
                            future_reserve_bytes=0, max_source_bytes=167772160,
                            max_archive_bytes=201326592, max_members=8192,
                            max_metadata_bytes=4194304, max_evidence_bytes=8388608)
        (self.raw / 'supervisor-terminal.json').write_text(json.dumps(dict(status='FAILED_NO_RETRY', returncode=1, allocation={'SLURM_JOB_ID':'12345'})))
        (self.raw / 'intent.json').write_text(json.dumps(dict(policy=policy, scope='own_cpu32', output_root=str(self.raw), supervisor_context={'release_sha256':'abc'})))
        self.request['cpu_supervisor_release'] = {'sha256':'abc'}
        self.request['scheduler_identity_capture'] = dict(job_id='12345', accounting_job_id='12345',
            capture=dict(argv=['scontrol','show','job','12345','-o'], returncode=0,
                         stdout='JobId=12345 JobState=FAILED', stderr=''))
        self.backup = self.root / 'backup'
        self.backup.mkdir()
        patch = mock.patch.object(store, 'BACKUP_ROOT', self.backup)
        patch.start(); self.addCleanup(patch.stop)
        self.contract = store.load_modules()['contract']
        patch = mock.patch.object(self.contract, 'validate_paths', side_effect=lambda r: {'policy-tree':Path(r['source_root']), 'private-tree':Path(r['private_root'])})
        patch.start(); self.addCleanup(patch.stop)
        self.destination = self.backup / policy['policy_id'] / self.raw.name / 'job-12345'
        self.destination.parent.mkdir(parents=True)
        self.deadline = time.monotonic()+30

    def inventory(self, request):
        nodes = {}
        for prefix, root in [('policy-tree', self.raw), ('private-tree', self.private)]:
            for path in [root]+sorted(root.rglob('*')):
                info = path.lstat()
                row = dict(source_path=str(path), mode=stat.S_IMODE(info.st_mode), uid=info.st_uid,
                           gid=info.st_gid, mtime_ns=info.st_mtime_ns)
                if path.is_symlink(): row.update(type='symlink', target=os.readlink(path))
                elif path.is_dir(): row.update(type='directory')
                else:
                    body=path.read_bytes()
                    row.update(type='file', bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
                nodes[prefix + (('/'+path.relative_to(root).as_posix()) if path!=root else '')]=row
        capture=request['scheduler_identity_capture']
        identity=self.contract.accounting_identity(request['job_id'],capture['accounting_job_id'],request['job_id'],capture['capture'])
        return dict(nodes=nodes, policy_id=request['policy_id'],scope=request['scope'],source_root=request['source_root'],
                    private_root=request['private_root'],job=dict(job_id=request['job_id'], terminal=True, state=('COMPLETED' if request['guard_status']=='RAW_REVIEW_REQUIRED' else 'FAILED'), exit_code=('0:0' if request['guard_status']=='RAW_REVIEW_REQUIRED' else '1:0'),identity=identity))

    def create(self, **kwargs):
        return store.create_archive(self.request, self.destination, inventory_reader=kwargs.pop('inventory_reader', self.inventory),
                                    free_reader=kwargs.pop('free_reader', lambda: 10**12), deadline=kwargs.pop('deadline', self.deadline), **kwargs)

    def test_complete_archive_preserves_originals(self):
        original = {str(p): p.read_bytes() for p in (self.raw/'intent.json', self.private/'cache')}
        result = self.create()
        self.assertEqual(result['status'], 'COMPLETE')
        archive=self.destination/'terminal-policy-full.tar.gz'
        self.assertEqual(result['archive_sha256'], hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertEqual(result['archive_bytes'], archive.stat().st_size)
        self.assertEqual(stat.S_IMODE(archive.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.destination.stat().st_mode), 0o700)
        verified=json.loads((self.destination/'verification.json').read_bytes())
        self.assertIn('policy-tree/empty', verified['origin_map'])
        self.assertEqual(verified['origin_map']['private-tree/alias']['target'], '/unread/target')
        self.assertLess(len(json.dumps(result)), 8192)
        for name,ref in result['controls'].items():
            body=Path(ref['path']).read_bytes()
            self.assertEqual(ref['bytes'], len(body))
            self.assertEqual(ref['sha256'], hashlib.sha256(body).hexdigest())
        for path, body in original.items(): self.assertEqual(Path(path).read_bytes(), body)
        for name in ('before.json','after.json','verification.json','index.json'):
            self.assertTrue((self.destination/name).is_file(), name)

    def set_xattr(self, path, value):
        if hasattr(os,'setxattr'):
            attr='user.z1_test'
            os.setxattr(path,attr,value)
        else:
            import ctypes
            libc=ctypes.CDLL(None,use_errno=True)
            attr='com.example.z1_test'
            self.assertEqual(libc.setxattr(os.fsencode(path),attr.encode(),value,len(value),0,0),0)
        return attr

    def success_records(self):
        self.request.update(guard_status='RAW_REVIEW_REQUIRED',guard_returncode=0)
        terminal=dict(status='RAW_REVIEW_REQUIRED',returncode=0,allocation={'SLURM_JOB_ID':'12345'})
        (self.raw/'supervisor-terminal.json').write_text(json.dumps(terminal))
        (self.raw/'worker-result.json').write_text(json.dumps(dict(status='RAW_REVIEW_REQUIRED',attempts=32)))

    def test_successful_cpu_archive_validates_actual_worker_result(self):
        self.success_records()
        self.assertEqual(self.create()['status'],'COMPLETE')

    def test_successful_cpu_archive_rejects_wrong_population(self):
        self.success_records()
        (self.raw/'worker-result.json').write_text(json.dumps(dict(status='RAW_REVIEW_REQUIRED',attempts=160)))
        with self.assertRaisesRegex(ValueError,'population'): self.create()
        self.assertFalse((self.destination/'index.json').exists())

    def test_successful_cpu_archive_requires_worker_result(self):
        self.success_records()
        (self.raw/'worker-result.json').unlink()
        with self.assertRaisesRegex(ValueError,'application record'): self.create()
        self.assertFalse((self.destination/'index.json').exists())

    def test_control_hash_and_exclusive_creation(self):
        path=self.backup/'control.json'
        data=b'{"status":"prepared"}'
        result=store.publish_control(path, data, hashlib.sha256(data).hexdigest(), free_reader=lambda:10**12, deadline=self.deadline)
        self.assertEqual(result['bytes'], len(data))
        self.assertEqual(path.read_bytes(), data)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        with self.assertRaises(FileExistsError):
            store.publish_control(path, data, hashlib.sha256(data).hexdigest(), free_reader=lambda:10**12, deadline=self.deadline)
        with self.assertRaisesRegex(ValueError, 'SHA'):
            store.publish_control(self.backup/'wrong.json', data, '0'*64, free_reader=lambda:10**12, deadline=self.deadline)
        self.assertFalse((self.backup/'wrong.json').exists())

    def test_control_higher_future_reserve(self):
        path=self.backup/'control.json'; data=b'x'
        with self.assertRaisesRegex(ValueError, 'reserve'):
            store.publish_control(path, data, hashlib.sha256(data).hexdigest(), free_reader=lambda:10**12,
                                  deadline=self.deadline, future_reserve_bytes=10**13)
        self.assertFalse(path.exists())

    def test_control_bad_future_reserve(self):
        for reserve in (0, -1, True, float('inf')):
            with self.subTest(reserve=reserve), self.assertRaises(ValueError):
                store.publish_control(self.backup/'c', b'x', hashlib.sha256(b'x').hexdigest(), free_reader=lambda:10**12,
                                      deadline=self.deadline, future_reserve_bytes=reserve)

    def test_oversized_control_rejected_before_hashing(self):
        data=b'x'*(store.METADATA_CAP+1)
        real=store.hashlib.sha256
        def checked(value=b''):
            if value is data: raise AssertionError('oversized data was hashed')
            return real(value)
        with mock.patch.object(store.hashlib,'sha256',side_effect=checked):
            with self.assertRaisesRegex(ValueError,'bounded'):
                store.publish_control(self.backup/'c',data,'0'*64,free_reader=lambda:10**12,deadline=self.deadline)
        self.assertFalse((self.backup/'c').exists())

    def test_control_outside_namespace(self):
        with self.assertRaisesRegex(ValueError, 'namespace'):
            store.publish_control(self.root/'outside', b'x', hashlib.sha256(b'x').hexdigest(), free_reader=lambda:10**12, deadline=self.deadline)

    def test_new_namespace_parents_created_exclusively(self):
        self.destination.parent.rmdir()
        self.destination.parent.parent.rmdir()
        self.backup.rmdir()
        result=self.create()
        self.assertEqual(result['created_parents'],[str(self.backup),str(self.destination.parent.parent),str(self.destination.parent)])
        for path in result['created_parents']:
            self.assertEqual(stat.S_IMODE(Path(path).stat().st_mode),0o700)

    def test_parent_file_conflict_is_preserved(self):
        self.destination.parent.rmdir()
        self.destination.parent.write_bytes(b'original')
        with self.assertRaisesRegex(ValueError,'nonsymlink'): self.create()
        self.assertEqual(self.destination.parent.read_bytes(),b'original')

    def test_parent_creation_waits_for_terminal_validation(self):
        self.destination.parent.rmdir()
        self.destination.parent.parent.rmdir()
        self.backup.rmdir()
        def wrong(request):
            value=self.inventory(request); value['job']['state']='RUNNING'; return value
        with self.assertRaises(ValueError): self.create(inventory_reader=wrong)
        self.assertFalse(self.backup.exists())

    def test_directory_conflict_is_preserved(self):
        self.destination.mkdir()
        marker=self.destination/'preexisting'; marker.write_bytes(b'evidence')
        with self.assertRaises(FileExistsError): self.create()
        self.assertEqual(marker.read_bytes(), b'evidence')
        self.assertEqual(list(self.destination.iterdir()), [marker])

    def test_destinations_are_closed_and_separate(self):
        for destination in (self.root/'foreign', self.raw/'backup', self.private/'backup'):
            with self.subTest(path=destination), self.assertRaises(ValueError):
                store.validate_destination(self.request, destination)
        self.request['private_root']=str(self.raw)
        with self.assertRaisesRegex(ValueError, 'distinct'): self.create()

    def test_backup_must_be_separate_from_runtime(self):
        self.request['physical_runtime_root']=str(self.root)
        with self.assertRaisesRegex(ValueError, 'runtime'): self.create()
        self.assertFalse(self.destination.exists())

    def test_parent_symlink_rejected(self):
        alias=self.root/'backup-alias'; alias.symlink_to(self.backup, target_is_directory=True)
        with mock.patch.object(store, 'BACKUP_ROOT', alias):
            with self.assertRaisesRegex(ValueError, 'nonsymlink'):
                store.publish_control(alias/'c', b'x', hashlib.sha256(b'x').hexdigest(), free_reader=lambda:10**12, deadline=self.deadline)
        self.assertFalse((self.backup/'c').exists())

    def test_source_parent_symlink_rejected(self):
        alias=self.root/'run-alias'; alias.symlink_to(self.raw.parent, target_is_directory=True)
        self.request['source_root']=str(alias/self.raw.name)
        with self.assertRaisesRegex(ValueError, 'nonsymlink'): self.create()
        self.assertFalse(self.destination.exists())

    def test_source_file_replacement_is_rejected(self):
        calls=0
        def inventory(request):
            nonlocal calls
            result=self.inventory(request); calls+=1
            if calls==2:
                path=self.private/'cache'; data=path.read_bytes(); path.unlink(); path.write_bytes(data)
            return result
        with self.assertRaisesRegex(ValueError, 'identity|metadata'): self.create(inventory_reader=inventory)
        self.assertTrue((self.destination/'failure.json').is_file())
        self.assertTrue((self.destination/'terminal-policy-full.tar.gz').is_file())
        self.assertFalse((self.destination/'index.json').exists())

    def test_source_content_drift_is_rejected(self):
        calls=0
        def inventory(request):
            nonlocal calls
            calls+=1
            if calls==2: (self.private/'cache').write_bytes(b'different-data')
            return self.inventory(request)
        with self.assertRaisesRegex(ValueError, 'drift'): self.create(inventory_reader=inventory)
        self.assertFalse((self.destination/'index.json').exists())

    def test_source_hardlinks_are_independent_bodies(self):
        os.link(self.private/'cache', self.private/'hardlink')
        self.create()
        before=json.loads((self.destination/'before.json').read_bytes())
        a,b=(before['nodes']['private-tree/'+name] for name in ('cache','hardlink'))
        self.assertEqual(a['source_identity'], b['source_identity'])
        self.assertGreaterEqual(a['source_identity']['link_count'],2)
        with tarfile.open(self.destination/'terminal-policy-full.tar.gz') as archive:
            self.assertTrue(archive.getmember('private-tree/hardlink').isfile())
            self.assertFalse(archive.getmember('private-tree/hardlink').islnk())

    def test_restore_xattrs_and_exact_nanosecond_time(self):
        path=self.private/'cache'
        attr=self.set_xattr(path,b'original\x00attribute')
        stamp=1700000000123456789
        os.utime(path, ns=(stamp-100,stamp))
        self.create()
        before=json.loads((self.destination/'before.json').read_bytes())['nodes']['private-tree/cache']
        self.assertEqual(before['restore_metadata']['atime_ns'], stamp-100)
        self.assertEqual(before['restore_metadata']['xattrs'][attr]['sha256'],hashlib.sha256(b'original\x00attribute').hexdigest())
        with tarfile.open(self.destination/'terminal-policy-full.tar.gz') as archive:
            pax=archive.getmember('private-tree/cache').pax_headers
            self.assertEqual(pax['mtime'],'1700000000.123456789')
            self.assertEqual(pax['z1.mtime_ns'],str(stamp))
            self.assertEqual(json.loads(pax['z1.restore_metadata']), before['restore_metadata'])
        index=json.loads((self.destination/'index.json').read_bytes())
        self.assertIn('private-tree/cache',index['read_induced_atime_changes'])

    def test_source_xattr_drift_is_not_exempt(self):
        calls=0
        def inventory(request):
            nonlocal calls
            calls+=1
            if calls==2:
                self.set_xattr(self.private/'cache',b'changed-attribute')
            return self.inventory(request)
        with self.assertRaisesRegex(ValueError,'drift'): self.create(inventory_reader=inventory)

    def test_member_listing_bounded_before_full_materialization(self):
        self.request['max_members']=3
        with self.assertRaisesRegex(ValueError,'member cap'): self.create()
        self.assertFalse(self.destination.exists())

    def test_metadata_inventory_cap(self):
        self.request['max_metadata_bytes']=100
        with self.assertRaisesRegex(ValueError,'metadata'): self.create()
        self.assertFalse(self.destination.exists())

    def test_fsync_bad_descriptor_is_not_suppressed(self):
        with mock.patch.object(store.os,'fsync',side_effect=OSError(errno.EBADF,'invalid descriptor')):
            with self.assertRaises(OSError): store._fsync_dir(self.backup)

    def test_unsupported_special_source_is_rejected(self):
        os.mkfifo(self.private/'fifo')
        with self.assertRaisesRegex(ValueError,'special'): self.create()
        self.assertFalse((self.destination/'index.json').exists())

    def test_source_cap_prevents_archive(self):
        self.request['max_source_bytes']=1
        with self.assertRaisesRegex(ValueError, 'source byte cap'): self.create()
        self.assertFalse((self.destination/'terminal-policy-full.tar.gz').exists())

    def test_archive_cap_retains_partial_and_failure(self):
        self.request['max_archive_bytes']=20
        with self.assertRaisesRegex(ValueError, 'byte cap'): self.create()
        self.assertTrue((self.destination/'terminal-policy-full.tar.gz').exists())
        self.assertTrue((self.destination/'failure.json').exists())
        self.assertFalse((self.destination/'index.json').exists())

    def test_quota_is_checked_during_archive_writes(self):
        def free():
            archive=self.destination/'terminal-policy-full.tar.gz'
            return store.FUTURE_MIN+store.FLOOR_CONTROLS if archive.exists() else 10**12
        with self.assertRaisesRegex(ValueError,'quota'): self.create(free_reader=free)
        self.assertTrue((self.destination/'attempt.json').is_file())
        self.assertTrue((self.destination/'terminal-policy-full.tar.gz').exists())
        self.assertFalse((self.destination/'index.json').exists())

    def test_actual_enospc_retains_partial_control(self):
        real=store.os.write
        count=0
        def failing(fd,data):
            nonlocal count
            count+=1
            if count==1: return real(fd,data[:1])
            raise OSError(errno.ENOSPC,'synthetic quota exhaustion')
        path=self.backup/'control'
        with mock.patch.object(store.os,'write',side_effect=failing), self.assertRaises(OSError):
            store.publish_control(path,b'abc',hashlib.sha256(b'abc').hexdigest(),free_reader=lambda:10**12,deadline=self.deadline)
        self.assertEqual(path.read_bytes(),b'a')

    def test_short_writes_are_completed(self):
        real=store.os.write
        with mock.patch.object(store.os,'write',side_effect=lambda fd,data: real(fd,data[:3])):
            self.create()
        self.assertTrue((self.destination/'index.json').is_file())

    def test_archive_fsync_failure_preserves_body_without_index(self):
        real=store.os.fsync
        def fail(fd):
            path=self.destination/'terminal-policy-full.tar.gz'
            if path.exists() and os.fstat(fd).st_ino==path.stat().st_ino:
                raise OSError(errno.EIO,'archive fsync failure')
            return real(fd)
        with mock.patch.object(store.os,'fsync',side_effect=fail), self.assertRaises(OSError): self.create()
        self.assertGreater((self.destination/'terminal-policy-full.tar.gz').stat().st_size,0)
        self.assertTrue((self.destination/'failure.json').exists())
        self.assertFalse((self.destination/'index.json').exists())

    def test_index_fsync_failure_is_bound_evidence_without_pass(self):
        real=store.os.fsync
        def fail(fd):
            path=self.destination/'index.json'
            if path.exists() and os.fstat(fd).st_ino==path.stat().st_ino:
                raise OSError(errno.EIO,'index fsync failure')
            return real(fd)
        with mock.patch.object(store.os,'fsync',side_effect=fail), self.assertRaises(OSError): self.create()
        self.assertEqual(json.loads((self.destination/'index.json').read_bytes())['status'],'ARCHIVE_BOUND')
        self.assertTrue((self.destination/'failure.json').exists())

    def test_deadline_expiration_after_partial_archive(self):
        real=store.os.write
        now=time.monotonic(); expires=now+30
        clock=[now]
        def expiring(fd,data):
            count=real(fd,data)
            path=self.destination/'terminal-policy-full.tar.gz'
            if path.exists() and os.fstat(fd).st_ino==path.stat().st_ino: clock[0]=expires+1
            return count
        with mock.patch.object(store.time,'monotonic',side_effect=lambda:clock[0]), mock.patch.object(store.os,'write',side_effect=expiring):
            with self.assertRaises(TimeoutError): self.create(deadline=expires)
        self.assertTrue((self.destination/'terminal-policy-full.tar.gz').exists())
        self.assertFalse((self.destination/'index.json').exists())

    def test_zero_write_fails_and_preserves_partial(self):
        with mock.patch.object(store.os,'write',return_value=0), self.assertRaisesRegex(ValueError,'short write'):
            self.create()
        self.assertTrue((self.destination/'attempt.json').exists())
        self.assertFalse((self.destination/'index.json').exists())

    def test_fsync_failure_has_no_complete_index(self):
        real=store.os.fsync
        def fail(fd):
            if stat.S_ISREG(os.fstat(fd).st_mode): raise OSError(errno.EIO,'synthetic fsync failure')
            return real(fd)
        with mock.patch.object(store.os,'fsync',side_effect=fail), self.assertRaises(OSError): self.create()
        self.assertTrue((self.destination/'attempt.json').exists())
        self.assertFalse((self.destination/'index.json').exists())

    def test_expired_deadline_before_and_during_write(self):
        with self.assertRaises(TimeoutError): self.create(deadline=time.monotonic()-1)
        self.assertFalse(self.destination.exists())
        def slow_free():
            time.sleep(.003); return 10**12
        with self.assertRaises(TimeoutError): self.create(free_reader=slow_free,deadline=time.monotonic()+.001)
        self.assertFalse(self.destination.exists())

    def test_wrong_terminal_record_identity_fails(self):
        path=self.raw/'supervisor-terminal.json'; value=json.loads(path.read_bytes())
        value['allocation']['SLURM_JOB_ID']='999'; path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'allocation'): self.create()
        self.assertFalse((self.destination/'index.json').exists())

    def test_wrong_policy_intent_fails(self):
        path=self.raw/'intent.json'; value=json.loads(path.read_bytes())
        value['policy']['arm']='extended200'; path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'policy'): self.create()
        self.assertFalse((self.destination/'index.json').exists())

    def test_wrong_supervisor_release_fails(self):
        self.request['cpu_supervisor_release']['sha256']='foreign'
        with self.assertRaisesRegex(ValueError,'release'): self.create()

    def test_wrong_job_inventory_fails(self):
        def wrong(request):
            value=self.inventory(request); value['job']['job_id']='999'; return value
        with self.assertRaisesRegex(ValueError,'identity'): self.create(inventory_reader=wrong)

    def test_saved_scheduler_identity_must_match_inventory(self):
        def wrong(request):
            value=self.inventory(request); value['job']['identity']['job_id']='foreign'; return value
        with self.assertRaisesRegex(ValueError,'identity'): self.create(inventory_reader=wrong)

    def test_live_scheduler_job_fails(self):
        def wrong(request):
            value=self.inventory(request); value['job']['state']='RUNNING'; return value
        with self.assertRaisesRegex(ValueError,'terminal'): self.create(inventory_reader=wrong)

    def test_stream_back_only_complete_bound_archive(self):
        result=self.create()
        output=io.BytesIO()
        returned=store.stream_back(self.request,self.destination,output,expected_archive_sha256=result['archive_sha256'],expected_archive_bytes=result['archive_bytes'],expected_index_sha256=result['controls']['index']['sha256'],deadline=self.deadline)
        self.assertEqual(output.getvalue(),(self.destination/'terminal-policy-full.tar.gz').read_bytes())
        self.assertEqual(returned['archive_sha256'],result['archive_sha256'])
        (self.destination/'terminal-policy-full.tar.gz').write_bytes(b'corrupt')
        output=io.BytesIO()
        with self.assertRaises(ValueError):
            store.stream_back(self.request,self.destination,output,expected_archive_sha256=result['archive_sha256'],expected_archive_bytes=result['archive_bytes'],expected_index_sha256=result['controls']['index']['sha256'],deadline=self.deadline)
        self.assertEqual(output.getvalue(),b'')

    def test_stream_back_bounds_archive_growth_during_prehash(self):
        self.request['max_archive_bytes']=4096
        result=self.create()
        path=self.destination/'terminal-policy-full.tar.gz'
        archive_inode=path.stat().st_ino
        real_fdopen=store.os.fdopen
        reads=[]
        class GrowingArchive:
            def __init__(self, stream):
                self.stream=stream
                self.appended=False
            def __enter__(self):
                self.stream.__enter__()
                return self
            def __exit__(self,*args): return self.stream.__exit__(*args)
            def __getattr__(self,key): return getattr(self.stream,key)
            def read(self,size):
                block=self.stream.read(size)
                reads.append((size,len(block)))
                if not self.appended:
                    self.appended=True
                    with path.open('ab') as growing: growing.write(b'x'*8192)
                return block
        def fdopen(fd,*args,**kwargs):
            archive=os.fstat(fd).st_ino==archive_inode
            stream=real_fdopen(fd,*args,**kwargs)
            return GrowingArchive(stream) if archive else stream
        output=io.BytesIO()
        with mock.patch.object(store.os,'fdopen',side_effect=fdopen):
            with self.assertRaises(ValueError):
                store.stream_back(self.request,self.destination,output,
                    expected_archive_sha256=result['archive_sha256'],expected_archive_bytes=result['archive_bytes'],
                    expected_index_sha256=result['controls']['index']['sha256'],deadline=self.deadline)
        self.assertEqual(output.getvalue(),b'')
        self.assertLessEqual(sum(count for _,count in reads),result['archive_bytes']+1)
        consumed=0
        for requested,count in reads:
            self.assertLessEqual(requested,min(store.CHUNK,result['archive_bytes']-consumed+1))
            consumed+=count

    def test_stream_back_bounds_archive_growth_during_emission(self):
        self.request['max_archive_bytes']=4096
        result=self.create()
        path=self.destination/'terminal-policy-full.tar.gz'
        output=io.BytesIO()
        class GrowingSink:
            appended=False
            def write(self,data):
                count=output.write(data)
                if not self.appended:
                    self.appended=True
                    with path.open('ab') as growing: growing.write(b'x'*8192)
                return count
        with self.assertRaises(ValueError):
            store.stream_back(self.request,self.destination,GrowingSink(),
                expected_archive_sha256=result['archive_sha256'],expected_archive_bytes=result['archive_bytes'],
                expected_index_sha256=result['controls']['index']['sha256'],deadline=self.deadline)
        self.assertLessEqual(len(output.getvalue()),result['archive_bytes'])
        self.assertLessEqual(len(output.getvalue()),self.request['max_archive_bytes'])

    def test_stream_back_retains_final_descriptor_drift_check(self):
        result=self.create()
        path=self.destination/'terminal-policy-full.tar.gz'
        output=io.BytesIO()
        class ChangingSink:
            changed=False
            def write(self,data):
                count=output.write(data)
                if not self.changed:
                    self.changed=True
                    with path.open('r+b') as archive: archive.write(b'X')
                return count
        with self.assertRaisesRegex(ValueError,'drift during stream-back'):
            store.stream_back(self.request,self.destination,ChangingSink(),
                expected_archive_sha256=result['archive_sha256'],expected_archive_bytes=result['archive_bytes'],
                expected_index_sha256=result['controls']['index']['sha256'],deadline=self.deadline)
        self.assertEqual(len(output.getvalue()),result['archive_bytes'])

    def test_stream_back_ignores_only_read_induced_descriptor_atime(self):
        result=self.create()
        real=store.os.fstat
        class StrictAtime:
            def __init__(self,info): self.info=info
            def __getattr__(self,key): return getattr(self.info,key)
            def __eq__(self,other): return False  # Strict-atime filesystems change atime on every read.
        with mock.patch.object(store.os,'fstat',side_effect=lambda fd:StrictAtime(real(fd))):
            output=io.BytesIO()
            store.stream_back(self.request,self.destination,output,expected_index_sha256=result['controls']['index']['sha256'],
                              expected_archive_sha256=result['archive_sha256'],expected_archive_bytes=result['archive_bytes'],deadline=self.deadline)
        self.assertEqual(len(output.getvalue()),result['archive_bytes'])

    def test_stream_back_partial_directory_fails(self):
        self.destination.mkdir()
        (self.destination/'terminal-policy-full.tar.gz').write_bytes(b'partial')
        with self.assertRaises((ValueError,FileNotFoundError)):
            store.stream_back(self.request,self.destination,io.BytesIO(),expected_archive_sha256='0'*64,expected_archive_bytes=7,expected_index_sha256='0'*64,deadline=self.deadline)

    def test_stream_back_rejects_unbound_index_and_failure_marker(self):
        result=self.create()
        kwargs=dict(expected_archive_sha256=result['archive_sha256'],expected_archive_bytes=result['archive_bytes'],
                    expected_index_sha256='0'*64,deadline=self.deadline)
        with self.assertRaisesRegex(ValueError,'index'):
            store.stream_back(self.request,self.destination,io.BytesIO(),**kwargs)
        kwargs['expected_index_sha256']=result['controls']['index']['sha256']
        (self.destination/'failure.json').write_bytes(b'{"status":"FAILED_NO_RETRY"}')
        with self.assertRaisesRegex(ValueError,'failure'):
            store.stream_back(self.request,self.destination,io.BytesIO(),**kwargs)

    def test_before_after_job_capture_timestamp_is_not_identity_drift(self):
        def reader(request):
            value=self.inventory(request); value['job']['captured_epoch']=time.time(); return value
        self.create(inventory_reader=reader)

    def test_inspect_and_compare_source_interface(self):
        before=store.inspect_source(self.request,inventory_reader=self.inventory,deadline=self.deadline)
        after=store.inspect_source(self.request,inventory_reader=self.inventory,deadline=self.deadline)
        result=store.compare_source_inventories(before,after)
        self.assertEqual(result['status'],'STABLE')
        after=copy.deepcopy(after); after['nodes']['private-tree/cache']['source_identity']['ctime_ns']+=1
        with self.assertRaisesRegex(ValueError,'drift'): store.compare_source_inventories(before,after)


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(store)
        self.modules=store.load_modules()

    def test_loader_restores_alias_and_never_imports_runtime(self):
        sentinel=object()
        with mock.patch.dict(sys.modules,{'archive_contract':sentinel}):
            result=store.load_modules()
            self.assertIs(sys.modules['archive_contract'],sentinel)
        self.assertEqual(set(result),{'contract','policy','remote'})
        self.assertFalse(any(name in sys.modules for name in ('torch','mujoco','mujoco_warp')))

    def test_loader_rejects_nonregular_sources_before_read(self):
        import vega_policy_archive.compat as compat
        with tempfile.TemporaryDirectory() as tmp:
            fd=os.open(tmp,os.O_RDONLY|os.O_DIRECTORY)
            try:
                with mock.patch.object(compat.os,'open',side_effect=lambda *args,**kwargs:os.dup(fd)):
                    with self.assertRaisesRegex(ValueError,'regular'): compat.load_modules()
            finally: os.close(fd)

    def test_frozen_sources_exact_except_explicit_record_reader(self):
        root=Path(__file__).resolve().parents[3]
        original=root/'.superpowers/sdd/2026-10-23-z1-overnight-execution/evaluation-staged-integration-repair-proposal002/archive-helper-candidate'
        copied=Path(store.__file__).parent/'compat'
        for name in ('archive_contract.py','remote_policy_archive.py'):
            self.assertEqual((original/name).read_bytes(),(copied/name).read_bytes())
        text=(copied/'policy_archive.py').read_text()
        text=text.replace(', *, record_reader=None):', '):')
        text=text.replace('    reader = read_archive_json if record_reader is None else record_reader\n','')
        text=text.replace("actual=reader(archive_path,'policy-tree/'+relative", "actual=read_archive_json(archive_path,'policy-tree/'+relative")
        self.assertEqual(text,(original/'policy_archive.py').read_text())

    def test_record_reader_seam_and_legacy_default_match(self):
        policy=self.modules['policy']
        request=dict(scope='own_cpu32',guard_status='FAILED_NO_RETRY',guard_returncode=1,job_id='42',
                     policy_id='x',policy={'arm':'native50','training_seed':2026102301},bank=None,
                     source_root='/raw',cpu_supervisor_release={'sha256':'s'},max_evidence_bytes=8388608)
        values={'supervisor-terminal.json':dict(status='FAILED_NO_RETRY',returncode=1,allocation={'SLURM_JOB_ID':'42'}),
                'intent.json':dict(policy=request['policy'],scope='own_cpu32',output_root='/raw',supervisor_context={'release_sha256':'s'})}
        records={name:dict(sha256=hashlib.sha256(json.dumps(value).encode()).hexdigest(),value=value) for name,value in values.items()}
        origins={'policy-tree/'+name:dict(type='file',sha256=item['sha256']) for name,item in records.items()}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'archive.gz'
            with tarfile.open(path,'w:gz') as archive:
                for name,value in values.items():
                    body=json.dumps(value).encode(); item=tarfile.TarInfo('policy-tree/'+name); item.size=len(body)
                    archive.addfile(item,io.BytesIO(body))
            policy.validate_archived_application_records(request,{'records':records},origins,path,time.monotonic()+30)
            called=[]
            def reader(path,name,deadline,maximum): called.append(name); return values[name.removeprefix('policy-tree/')]
            policy.validate_archived_application_records(request,{'records':records},origins,None,time.monotonic()+30,record_reader=reader)
            self.assertEqual(set(called),set(origins))

    def test_actual_closed_layout_and_policy_guards(self):
        contract=self.modules['contract']; root='/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002'
        rows=[]
        for arm in contract.ARMS:
            for seed in contract.SEEDS:
                for scope,bank in [('own_cpu32',None)]+[('science160',bank) for bank in contract.BANKS]:
                    name=f'{arm}-seed{seed}'; leaf='own_cpu32-staged-integration002' if bank is None else f'science160-bank{bank}'
                    rows.append(dict(policy_id=name,scope=scope,bank=bank,run_root=f'{root}/runs/{name}/{leaf}',private_root=f'{root}/private/{name}/{leaf}'))
        layout=dict(schema='z1_horizon_cpu_archive_layout_v1',root=root,slots=rows)
        self.assertEqual(contract.validate_layout(layout),layout)
        request=dict(cpu_layout=layout,scope='own_cpu32',bank=None,policy_id='native50-seed2026102301',
                     policy=dict(policy_id='native50-seed2026102301',arm='native50',training_seed=2026102301,
                                 task=contract.BASE+'-Horizon-native50',native_window_s=.05),campaign_root=contract.CAMPAIGN,job_id='42')
        row=next(row for row in rows if row['policy_id']==request['policy_id'] and row['scope']=='own_cpu32')
        request.update(source_root=row['run_root'],private_root=row['private_root'])
        contract.validate_paths(request)
        for field,value in [('job_id','fake'),('source_root','/foreign'),('policy_id','foreign')]:
            changed=copy.deepcopy(request); changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): contract.validate_paths(changed)
        changed=copy.deepcopy(layout); changed['slots'].pop()
        with self.assertRaises(ValueError): contract.validate_layout(changed)
        changed=copy.deepcopy(layout); changed['root']='/foreign'
        with self.assertRaises(ValueError): contract.validate_layout(changed)

if __name__ == '__main__': unittest.main()
