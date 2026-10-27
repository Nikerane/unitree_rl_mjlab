"""Pure acceptance tests: no environment/model imports and no real scheduler."""
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import journal
import scheduler
import coordinator

class FakeAdapter:
    def __init__(self, fail=None):
        self.events = []
        self.fail = fail
        self.bad = False
    def _gate(self, operation, *args):
        self.events.append((operation, *args))
        if operation == self.fail:
            raise ValueError('gate rejected')
    def prepare_once(self, phase, context):
        self._gate('prepare', phase)
        for index in scheduler.INDICES[phase]:
            (Path(context['root'])/'evidence'/('slot%02d'%index)).mkdir(parents=True,exist_ok=False)
    def preflight_current(self, phase, index, context): self._gate('preflight', index)
    def reserve(self, phase, index, context):
        self._gate('reserve', index)
        return {'descriptor': {'index': index, 'nonce': context['nonce']},
                'ledger': {'all18': True}, 'reservation': {'index': index}}
    def resource_check(self, transaction, context):
        if self.bad: raise ValueError('quota or role cap exceeded')
        return {'observed': time.monotonic()}
    def terminal(self, phase, index, transaction, identity, observation, context):
        self._gate('terminal', index)
        return {'qualified': True, 'receipt': {'path': '/closed/receipt', 'sha256': 'f'*64}}
    def phase_complete(self, phase, context): self._gate('complete', phase)

class FakeScheduler:
    def __init__(self, states=('COMPLETED',), delay=0):
        self.calls = []
        self.states = list(states)
        self.delay = delay
        self.cancelled = False
        self.unknown = False
        self.empty = False
    def capability(self, phase, index, name, log):
        self.calls.append(('test', index)); return {'returncode': 0, 'complete_eof': True}
    def controller(self):
        return {'job_id': '900', 'user': str(os.getuid()), 'state': 'RUNNING',
                'end_epoch': time.time()+28800, 'cpus': 2, 'memory': '2048M'}
    def submit(self, phase, index, name, log, *, permit):
        self.calls.append(('submit', index))
        if self.unknown: raise TimeoutError('unknown submission')
        return {'job_id': '123', 'name': name, 'index': index,
                'raw': {'argv': ['sbatch'], 'returncode': 0, 'stdout': '123\n', 'stderr': '', 'complete_eof': True}}
    def observe(self, submission):
        time.sleep(self.delay)
        if self.empty: raise ValueError('empty accounting')
        state = 'CANCELLED' if self.cancelled else self.states.pop(0) if len(self.states)>1 else self.states[0]
        ident = {'job_id': '125', 'array_job_id': '123', 'index': submission['index'],
                 'name': submission['name'], 'user': str(os.getuid())}
        return {'identity': ident, 'state': state, 'terminal': state in scheduler.TERMINAL,
                'raw': [{'argv': ['scontrol'], 'returncode': 0, 'stdout': state, 'stderr': '', 'complete_eof': True}]}
    def cancel(self, identity):
        self.calls.append(('cancel', identity['job_id'])); self.cancelled = True
        return {'argv': ['scancel', identity['job_id']], 'returncode': 0, 'stdout': '', 'stderr': '', 'complete_eof': True}

class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.context = {'nonce': 'a'*32, 'controller_job_id': '900', 'uid': str(os.getuid()),
                        'source_sha256': 'b'*64, 'config_sha256': 'c'*64}
    def tearDown(self): self.tmp.cleanup()
    def run_owner(self, adapter=None, sched=None, **kw):
        cfg = coordinator.Runtime(self.root, dict(self.context,log_root=str(self.root/'evidence')))
        return coordinator.run(cfg, adapter or FakeAdapter(), sched or FakeScheduler(), **kw)

class JournalTests(Fixture):
    def test_exclusive_lease_race_and_stale_never_reclaimed(self):
        def take(_):
            try: return journal.acquire_new_lease(self.root, self.context)
            except journal.Hold: return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            leases = list(pool.map(take, range(8)))
        self.assertEqual(sum(x is not None for x in leases), 1)
        winner = next(x for x in leases if x)
        winner.close()
        with self.assertRaises(journal.Hold): journal.acquire_new_lease(self.root, self.context)
    def test_chain_fsync_torn_and_modified_are_rejected(self):
        with patch.object(os, 'fsync', wraps=os.fsync) as sync:
            j = journal.Journal(self.root/'journal.jsonl')
            j.append({'kind': 'ACTIVATE'})
            j.append({'kind': 'SUBMIT_INTENT', 'index': 3})
            self.assertGreaterEqual(sync.call_count, 3)
        self.assertEqual(len(journal.read_chain(j.path)), 2)
        original = j.path.read_bytes()
        j.path.write_bytes(original[:-2])
        with self.assertRaises(journal.Hold): journal.read_chain(j.path)
        j.path.write_bytes(original.replace(b'ACTIVATE', b'DEACTIV8'))
        with self.assertRaises(journal.Hold): journal.read_chain(j.path)
    def test_reconcile_is_read_only_and_restart_denies_submit(self):
        self.run_owner()
        before = {p.name:p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        self.assertEqual(coordinator.reconcile(self.root)['status'], 'READ_ONLY')
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.root.iterdir() if p.is_file()})
        s = FakeScheduler()
        with self.assertRaises(journal.Hold): self.run_owner(sched=s)
        self.assertEqual(s.calls,[])

class OwnerTests(Fixture):
    def test_exact_order_one_slot_current_preflight_and_prepare_once(self):
        a, s = FakeAdapter(), FakeScheduler()
        result = self.run_owner(a, s)
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual([c[1] for c in s.calls if c[0]=='submit'], [3,6,9,12,15,1,2,4,5,7,8,10,11,13,14,16,17])
        self.assertEqual([e for e in a.events if e[0]=='prepare'], [('prepare','remaining'),('prepare','science')])
        self.assertEqual(len([e for e in a.events if e[0]=='preflight']),17)
        events=journal.read_chain(self.root/'journal.jsonl')
        kinds=[e['event']['kind'] for e in events]
        self.assertLess(kinds.index('RESERVATION'), kinds.index('SUBMIT_INTENT'))
        self.assertLess(kinds.index('SUBMIT_INTENT'), kinds.index('SUBMIT_RESULT'))
    def test_failed_gate_and_failed_child_block_every_later_slot(self):
        for gate in ['prepare','preflight','reserve','terminal','complete']:
            with self.subTest(gate=gate), tempfile.TemporaryDirectory() as d:
                s=FakeScheduler(); result=coordinator.run(coordinator.Runtime(Path(d), dict(self.context,log_root=str(Path(d)/'evidence'))), FakeAdapter(gate), s)
                self.assertEqual(result['status'],'HOLD')
                self.assertLessEqual(len([c for c in s.calls if c[0]=='submit']),5 if gate=='complete' else 1)
        s=FakeScheduler(('FAILED',)); result=self.run_owner(sched=s)
        self.assertEqual(result['status'],'HOLD')
        self.assertEqual([c for c in s.calls if c[0]=='submit'], [('submit',3)])
    def test_unknown_submit_never_replayed_even_without_identity(self):
        s=FakeScheduler(); s.unknown=True
        self.assertEqual(self.run_owner(sched=s)['status'],'HOLD')
        self.assertEqual([c for c in s.calls if c[0]=='submit'],[('submit',3)])
        self.assertFalse(any(c[0]=='cancel' for c in s.calls))
        with self.assertRaises(journal.Hold): self.run_owner(sched=s)
    def test_empty_accounting_holds_and_never_guesses_cancel_id(self):
        s=FakeScheduler(); s.empty=True
        self.assertEqual(self.run_owner(sched=s)['status'],'HOLD')
        self.assertFalse(any(c[0]=='cancel' for c in s.calls))
    def test_resource_breach_during_slow_observation_cancels_once_and_drains(self):
        a=FakeAdapter(); s=FakeScheduler(('RUNNING',),delay=.4)
        threading.Timer(.1, lambda:setattr(a,'bad',True)).start()
        start=time.monotonic(); result=self.run_owner(a,s)
        self.assertEqual(result['status'],'HOLD')
        self.assertLess(result['halt_observed_at']-start,.35)
        self.assertEqual([c for c in s.calls if c[0]=='cancel'],[('cancel','125')])
        self.assertEqual([c for c in s.calls if c[0]=='submit'],[('submit',3)])
        self.assertTrue(result['drained'])
    def test_low_issue_budget_has_no_submit_and_signal_has_no_late_issue(self):
        s=FakeScheduler(); s.controller=lambda:{'job_id':'900','user':str(os.getuid()),'state':'RUNNING','end_epoch':time.time()+3629,'cpus':2,'memory':'2048M'}
        self.assertEqual(self.run_owner(sched=s)['status'],'HOLD')
        self.assertFalse(any(c[0]=='submit' for c in s.calls))
        with tempfile.TemporaryDirectory() as d:
            stop=threading.Event(); stop.set(); s=FakeScheduler()
            self.assertEqual(coordinator.run(coordinator.Runtime(Path(d),dict(self.context,log_root=str(Path(d)/'evidence'))),FakeAdapter(),s,stop=stop)['status'],'HOLD')
            self.assertFalse(any(c[0]=='submit' for c in s.calls))
    def test_pending_and_active_deadline_one_cancel(self):
        for mode in ('pending','active'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                s=FakeScheduler(('PENDING' if mode=='pending' else 'RUNNING',))
                class Clock:
                    value=0
                    def __call__(self):
                        self.value += 301 if mode=='pending' else 600
                        return self.value
                r=coordinator.run(coordinator.Runtime(Path(d),dict(self.context,log_root=str(Path(d)/'evidence'))),FakeAdapter(),s,clock=Clock())
                self.assertEqual(r['status'],'HOLD')
                self.assertEqual(len([c for c in s.calls if c[0]=='cancel']),1)
    def test_unknown_cancel_stays_hold(self):
        a=FakeAdapter(); s=FakeScheduler(('RUNNING',),delay=.15)
        def cancel(identity):
            s.calls.append(('cancel',identity['job_id'])); raise TimeoutError('unknown cancellation')
        s.cancel=cancel
        threading.Timer(.05, lambda:setattr(a,'bad',True)).start()
        r=self.run_owner(a,s)
        self.assertEqual(r['status'],'HOLD'); self.assertFalse(r['drained'])
        self.assertEqual(len([c for c in s.calls if c[0]=='cancel']),1)
    def test_nested_capability_failure_before_lease(self):
        s=FakeScheduler(); s.capability=lambda *a:{'returncode':1,'complete_eof':True}
        with self.assertRaises(journal.Hold): self.run_owner(sched=s)
        self.assertFalse((self.root/'owner.lease').exists())

class TransportTests(Fixture):
    def test_real_child_exit_eof_late_failure_and_bounded_output(self):
        r=scheduler.run_process([sys.executable,'-c','import sys;print("PASS");sys.exit(1)'],timeout=2)
        self.assertEqual(r['returncode'],1); self.assertTrue(r['complete_eof'])
        self.assertEqual(r['stdout'],'PASS\n'); self.assertGreater(r['pid'],0)
        r=scheduler.run_process([sys.executable,'-c','print("x"*100000)'],timeout=2,cap=32)
        self.assertEqual(r['outcome'],'UNKNOWN'); self.assertLessEqual(len(r['stdout']),32)
    def test_transport_survives_client_pipe_disconnect(self):
        marker=self.root/'done'
        runner=self.root/'runner.py'
        runner.write_text('import sys\nsys.path.insert(0,'+repr(str(ROOT))+')\nimport scheduler\nscheduler.run_process([sys.executable,"-c",'+repr('from pathlib import Path;import time;time.sleep(.2);Path('+repr(str(marker))+').write_text("done")')+'],timeout=2)\n')
        p=subprocess.Popen([sys.executable,'-B',str(runner)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        p.stdout.close(); p.stderr.close()
        self.assertEqual(p.wait(timeout=4),0); self.assertEqual(marker.read_text(),'done')
    def test_environment_and_policy_argv_keep_resources_in_batch(self):
        env=scheduler.child_environment({'SLURM_JOB_ID':'900','SLURM_STEP_ID':'0','SBATCH_MEM':'1','PYTHONPATH':'bad','PYTHONHOME':'bad','PATH':'/bin','HOME':'/home/x'})
        self.assertEqual(env,{'PATH':'/bin','HOME':'/home/x'})
        argv=scheduler.policy_argv('remaining',3,'z1a005-'+'a'*32+'-03','/closed/log',test=False)
        self.assertIn('--array=3%1',argv); self.assertFalse(any('cpus' in a or '--mem' in a for a in argv))
        with self.assertRaises(journal.Hold): scheduler.policy_argv('remaining',0,'x','/closed/log',test=False)
    def test_materialized_identity_cross_checks_both_sources(self):
        record={'job_id':'123','index':3,'name':'z1a005-'+'a'*32+'-03'}
        control='JobId=125 ArrayJobId=123 ArrayTaskId=3 JobName='+record['name']+' UserId=u('+str(os.getuid())+') JobState=RUNNING'
        acct='125|'+record['name']+'|'+str(os.getuid())+'|RUNNING|2026-01-01T01:00:00|Unknown|Unknown|0:0\n'
        identity=scheduler.materialize(record,control,acct,str(os.getuid()))
        self.assertEqual(identity['job_id'],'125')
        for bad in (control.replace('ArrayTaskId=3','ArrayTaskId=6'),control.replace('JobId=125','JobId=126',1)):
            with self.assertRaises(journal.Hold): scheduler.materialize(record,bad,acct,str(os.getuid()))


class AdditionalBoundaries(Fixture):
    def test_reservation_reference_is_actual_reservation_not_wrapper(self):
        self.run_owner()
        frozen=json.loads((self.root/'slot03-reservation.json').read_text())
        self.assertEqual(frozen, {'index':3})
        self.assertTrue((self.root/'slot03-ledger.json').exists())
        self.assertTrue((self.root/'slot03-descriptor.json').exists())
    def test_duplicate_intent_cannot_be_durable_twice(self):
        j=journal.Journal(self.root/'journal.jsonl')
        j.append({'kind':'SUBMIT_INTENT','index':3})
        with self.assertRaises(journal.Hold): j.append({'kind':'SUBMIT_INTENT','index':3})
        self.assertEqual(len(journal.read_chain(j.path)),1)
    def test_controller_end_shortening_causes_one_owned_drain(self):
        s=FakeScheduler(('RUNNING','COMPLETED')); calls=[0]
        def controller():
            calls[0]+=1
            return {'job_id':'900','user':str(os.getuid()),'state':'RUNNING',
                    'end_epoch':time.time()+(28800 if calls[0]==1 else 1500),'cpus':2,'memory':'2048M'}
        s.controller=controller
        r=self.run_owner(sched=s)
        self.assertEqual(r['status'],'HOLD')
        self.assertTrue(r['drained'])
        self.assertEqual([c for c in s.calls if c[0]=='cancel'],[('cancel','125')])
    def test_lease_race_in_distinct_processes(self):
        code='import sys;sys.path.insert(0,'+repr(str(ROOT))+');import journal;from pathlib import Path\ntry:\n l=journal.acquire_new_lease(Path(sys.argv[1]),{});print("OWNED",flush=True)\nexcept journal.Hold:\n print("DENIED",flush=True)\n'
        children=[subprocess.Popen([sys.executable,'-B','-c',code,str(self.root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(6)]
        outputs=[p.communicate(timeout=5)[0].decode().strip() for p in children]
        self.assertEqual(outputs.count('OWNED'),1)
        self.assertEqual(outputs.count('DENIED'),5)

class LiveResourceTests(Fixture):
    def test_actual_log_and_quota_growth_during_real_child_wait_are_observed(self):
        for resource in ('log','quota','raw'):
            with self.subTest(resource=resource), tempfile.TemporaryDirectory() as d:
                root=Path(d); owned=root/'usage'; owned.write_bytes(b'')
                quota=root/'quota'; quota.write_text('100')
                class Adapter(FakeAdapter):
                    def __init__(self): super().__init__(); self.observed=[]
                    def resource_check(self,transaction,context):
                        self.observed.append(time.monotonic())
                        if owned.stat().st_size>10 or int(quota.read_text())<50: raise ValueError(resource+' breach')
                        return {'bytes':owned.stat().st_size}
                observation_entered=threading.Event(); mutated=threading.Event(); errors=[]
                class SlowScheduler(FakeScheduler):
                    def observe(self,submission):
                        # Barrier is inside the issued observation, never preissue.
                        observation_entered.set()
                        raw=scheduler.run_process([sys.executable,'-c','import time;time.sleep(.35)'],timeout=2)
                        result=super().observe(submission); result['raw'].append(raw); return result
                a=Adapter(); s=SlowScheduler(('RUNNING',))
                def grow():
                    try:
                        if not observation_entered.wait(timeout=5): return
                        if resource=='quota':
                            pending=root/'quota-next'; pending.write_text('0'); pending.replace(quota)
                        else: owned.write_bytes(b'x'*11)
                        mutated.set()
                    except BaseException as exc: errors.append(repr(exc))
                mutation=threading.Thread(target=grow)
                mutation.start()
                try:
                    r=coordinator.run(coordinator.Runtime(root,dict(self.context,log_root=str(root/'evidence'))),a,s)
                finally:
                    mutation.join(timeout=6)
                diagnostic=repr({'result':r,'calls':s.calls,'resource_times':a.observed,
                                 'observation_entered':observation_entered.is_set(),'mutated':mutated.is_set(),'errors':errors})
                self.assertFalse(mutation.is_alive(),diagnostic)
                self.assertFalse(errors,diagnostic)
                self.assertTrue(mutated.is_set(),diagnostic)
                self.assertEqual(r['status'],'HOLD',diagnostic)
                self.assertEqual(r['reason'],'resource gate: '+resource+' breach',diagnostic)
                self.assertTrue(r['drained'],diagnostic)
                self.assertEqual([c for c in s.calls if c[0] in ('submit','cancel')],[('submit',3),('cancel','125')],diagnostic)
                self.assertLess(max(y-x for x,y in zip(a.observed,a.observed[1:])),.25,diagnostic)
    def test_preissue_resource_breach_holds_without_submit_or_cancel(self):
        adapter=FakeAdapter(); adapter.bad=True; sched=FakeScheduler()
        result=self.run_owner(adapter,sched)
        diagnostic=repr({'result':result,'calls':sched.calls})
        self.assertEqual(result['status'],'HOLD',diagnostic)
        self.assertEqual(result['reason'],'resource gate: quota or role cap exceeded',diagnostic)
        self.assertFalse(result['drained'],diagnostic)
        self.assertEqual([c for c in sched.calls if c[0] in ('submit','cancel')],[],diagnostic)
    def test_real_usr1_during_preissue_gate_prevents_submission(self):
        class Adapter(FakeAdapter):
            def reserve(self,*args):
                threading.Timer(.03,lambda:os.kill(os.getpid(),signal.SIGUSR1)).start()
                time.sleep(.15)
                return super().reserve(*args)
        s=FakeScheduler(); r=self.run_owner(Adapter(),s)
        self.assertEqual(r['status'],'HOLD')
        self.assertFalse(any(c[0]=='submit' for c in s.calls))
    def test_bounded_journal_exhaustion_drains_existing_child(self):
        s=FakeScheduler(('RUNNING','COMPLETED'),delay=.15)
        real_append=journal.Journal.append
        triggered=[False]
        def bounded(j,event,**kwargs):
            if event['kind']=='OBSERVATION' and not triggered[0]:
                triggered[0]=True
                journal.REGULAR_JOURNAL=1
            return real_append(j,event,**kwargs)
        with patch.object(journal,'REGULAR_JOURNAL',8388608,create=True):
            with patch.object(journal.Journal,'append',bounded): r=self.run_owner(sched=s)
        self.assertEqual(r['status'],'HOLD')
        self.assertEqual([c for c in s.calls if c[0]=='cancel'],[('cancel','125')])
        self.assertTrue(r['drained'])

class FailureEvidenceTests(Fixture):
    def test_binary_scheduler_output_retains_exact_bytes_and_timeout_evidence(self):
        import base64
        r=scheduler.run_process([sys.executable,'-c','import os;os.write(1,b"\\xff\\x00")'],timeout=2)
        self.assertEqual(base64.b64decode(r['stdout_base64']),b'\xff\x00')
        self.assertEqual(r['stdout_bytes'],2)
        r=scheduler.run_process([sys.executable,'-c','import time;print("partial",flush=True);time.sleep(4)'],timeout=.1)
        self.assertEqual(r['outcome'],'UNKNOWN'); self.assertFalse(r['complete_eof'])
        self.assertIn('partial',r['stdout'])
    def test_hash_valid_duplicate_intent_is_not_reconciled_as_clean(self):
        j=journal.Journal(self.root/'journal.jsonl')
        j.append({'kind':'SUBMIT_INTENT','index':3})
        body={'sequence':2,'previous':j.previous,'event':{'kind':'SUBMIT_INTENT','index':3}}
        row=dict(body,sha256=journal.digest(body))
        with j.path.open('ab') as out: out.write(journal.encoded(row)+b'\n')
        with self.assertRaises(journal.Hold): journal.read_chain(j.path)

class BoundBatchTests(Fixture):
    def test_changed_batch_is_rejected_before_transport(self):
        batch=self.root/'remaining.sbatch'; batch.write_bytes(b'unchanged physical batch')
        refs={phase:{'path':str(self.root/(phase+'.sbatch')),'sha256':__import__('hashlib').sha256(b'unchanged physical batch').hexdigest()} for phase in ('remaining','science')}
        (self.root/'science.sbatch').write_bytes(b'unchanged physical batch')
        with patch.dict(scheduler.BATCHES,{p:r['path'] for p,r in refs.items()}):
            sched=scheduler.Scheduler('900',str(os.getuid()),refs)
            batch.write_bytes(b'changed worker arguments')
            with patch.object(scheduler,'run_process',side_effect=AssertionError('must reject before spawning')):
                with self.assertRaises(journal.Hold): sched.submit('remaining',3,'z1a005-'+'a'*32+'-03',str(self.root),permit=None)

class LateIssueTests(Fixture):
    def test_revoked_or_expired_permit_never_starts_scheduler_child(self):
        stop=threading.Event(); stop.set()
        permit=scheduler.IssuePermit(stop,time.monotonic()+5)
        with patch.object(subprocess,'Popen',side_effect=AssertionError('late child spawn')):
            r=scheduler.run_process(['/usr/bin/sbatch'],timeout=2,permit=permit)
        self.assertEqual(r['outcome'],'NOT_STARTED')
        permit=scheduler.IssuePermit(threading.Event(),time.monotonic()-1)
        with patch.object(subprocess,'Popen',side_effect=AssertionError('expired child spawn')):
            r=scheduler.run_process(['/usr/bin/sbatch'],timeout=2,permit=permit)
        self.assertEqual(r['outcome'],'NOT_STARTED')

class LifecycleReviewTests(Fixture):
    def test_terminal_preservation_crossing1530_completes_slot(self):
        class Clock:
            value=0
            def __call__(self): return self.value
        clock=Clock()
        class BudgetScheduler(FakeScheduler):
            def __init__(self): super().__init__(('PENDING','RUNNING','COMPLETED'))
            def controller(self):
                return dict(super().controller(),end_epoch=time.time()+3635-clock.value)
            def observe(self,submission):
                result=super().observe(submission)
                if result['state']=='PENDING': clock.value=899
                elif result['state']=='COMPLETED': clock.value=2099
                return result
        class Adapter(FakeAdapter):
            def terminal(self,*args):
                self.processing_deadline=args[-1]['deadline']
                clock.value=2109
                return super().terminal(*args)
        adapter=Adapter(); sched=BudgetScheduler()
        with patch.object(coordinator,'SCHEDULER_INTERVAL',0):
            result=self.run_owner(adapter,sched,clock=clock)
        rows=journal.read_chain(self.root/'journal.jsonl')
        completed=[r['event']['index'] for r in rows if r['event']['kind']=='SLOT_COMPLETE']
        diagnostic=repr({'result':result,'calls':sched.calls,'completed':completed})
        self.assertEqual(completed,[3],diagnostic)
        self.assertEqual(result['reason'],'issue lifetime below3630',diagnostic)
        self.assertEqual([c for c in sched.calls if c[0] in ('submit','cancel')],[('submit',3)],diagnostic)
        self.assertAlmostEqual(adapter.processing_deadline,3299,delta=.05)

    def test_terminal_postprocessing_must_finish_before_its_clamped_deadline(self):
        class Clock:
            value=0
            def __call__(self): return self.value
        clock=Clock()
        class Adapter(FakeAdapter):
            def terminal(self,*args):
                clock.value=args[-1]['deadline']+1
                return super().terminal(*args)
        sched=FakeScheduler()
        result=self.run_owner(Adapter(),sched,clock=clock)
        rows=journal.read_chain(self.root/'journal.jsonl')
        diagnostic=repr({'result':result,'calls':sched.calls})
        self.assertEqual(result['status'],'HOLD',diagnostic)
        self.assertIn('postprocessing deadline',result['reason'],diagnostic)
        self.assertFalse(any(r['event']['kind']=='SLOT_COMPLETE' for r in rows),diagnostic)
        self.assertEqual([c for c in sched.calls if c[0] in ('submit','cancel')],[('submit',3)],diagnostic)

    def test_quiescent_drain_paces_multiple_nonterminal_observations(self):
        stop=threading.Event(); drain_times=[]; resource_times=[]
        class Adapter(FakeAdapter):
            def resource_check(self,*args):
                resource_times.append(time.monotonic())
                return super().resource_check(*args)
        class DelayedCancelScheduler(FakeScheduler):
            def observe(self,submission):
                if not self.cancelled:
                    result=super().observe(submission)
                    stop.set()
                    return result
                drain_times.append(time.monotonic())
                self.cancelled=False
                result=super().observe(submission)
                self.cancelled=True
                result['state']='CANCELLED' if len(drain_times)==4 else 'COMPLETING'
                result['terminal']=result['state']=='CANCELLED'
                return result
        sched=DelayedCancelScheduler(('RUNNING',))
        with patch.object(coordinator,'SCHEDULER_INTERVAL',.12), patch.object(coordinator,'DRAIN_SECONDS',.8):
            result=self.run_owner(Adapter(),sched,stop=stop)
        rows=journal.read_chain(self.root/'journal.jsonl')
        gaps=[y-x for x,y in zip(drain_times,drain_times[1:])]
        first_quiesce=next(i for i,row in enumerate(rows) if row['event']['kind']=='QUIESCE')
        emergency_bytes=sum(len(line) for line in (self.root/'journal.jsonl').read_bytes().splitlines(keepends=True)[first_quiesce:])
        diagnostic=repr({'result':result,'calls':sched.calls,'gaps':gaps,'resources':resource_times,'emergency_bytes':emergency_bytes})
        self.assertEqual(len(drain_times),4,diagnostic)
        self.assertTrue(all(gap>=.10 for gap in gaps),diagnostic)
        self.assertTrue(result['drained'],diagnostic)
        self.assertEqual([c for c in sched.calls if c[0] in ('submit','cancel')],[('submit',3),('cancel','125')],diagnostic)
        self.assertLess(emergency_bytes,32768,diagnostic)
        self.assertEqual(rows[-1]['event']['kind'],'HOLD',diagnostic)
        self.assertLess(max(y-x for x,y in zip(resource_times,resource_times[1:])),.25,diagnostic)

    def test_drain_stops_when_refreshed_controller_reaches_finalization_boundary(self):
        stop=threading.Event(); drain_times=[]
        class ShortenedController(FakeScheduler):
            def controller(self):
                result=super().controller()
                if self.cancelled: result['end_epoch']=time.time()+100
                return result
            def observe(self,submission):
                if not self.cancelled:
                    result=super().observe(submission); stop.set(); return result
                drain_times.append(time.monotonic())
                self.cancelled=False
                result=super().observe(submission)
                self.cancelled=True
                result['state']='CANCELLED' if len(drain_times)==4 else 'COMPLETING'
                result['terminal']=result['state']=='CANCELLED'
                return result
        sched=ShortenedController(('RUNNING',))
        with patch.object(coordinator,'SCHEDULER_INTERVAL',.12), patch.object(coordinator,'DRAIN_SECONDS',.8):
            result=self.run_owner(sched=sched,stop=stop)
        rows=journal.read_chain(self.root/'journal.jsonl')
        diagnostic=repr({'result':result,'calls':sched.calls,'drain_times':drain_times})
        self.assertEqual(len(drain_times),1,diagnostic)
        self.assertFalse(result['drained'],diagnostic)
        self.assertEqual(result['status'],'HOLD',diagnostic)
        self.assertEqual([c for c in sched.calls if c[0] in ('submit','cancel')],[('submit',3),('cancel','125')],diagnostic)
        unknown=[r['event'] for r in rows if r['event']['kind']=='DRAIN_UNKNOWN']
        self.assertEqual([r['reason'] for r in unknown],['terminal drain deadline'],diagnostic)
        self.assertFalse(any(r['event']['kind'] in ('SLOT_COMPLETE','TERMINAL_PROCESSED') for r in rows),diagnostic)

class GuardianIOTests(Fixture):
    def test_resource_observation_continues_during_slow_durable_quiesce_and_cancel(self):
        times=[]; callback_threads=[]; append_threads=[]; slow_spans=[]
        callback_lock=threading.Lock(); concurrency=[0,0]
        class Adapter(FakeAdapter):
            def resource_check(self,*args):
                with callback_lock:
                    concurrency[0]+=1; concurrency[1]=max(concurrency)
                try:
                    times.append(time.monotonic()); callback_threads.append(threading.get_ident())
                    return super().resource_check(*args)
                finally:
                    with callback_lock: concurrency[0]-=1
        adapter=Adapter()
        class BreachScheduler(FakeScheduler):
            def observe(self,submission):
                adapter.bad=True
                return super().observe(submission)
        sched=BreachScheduler(('RUNNING',),delay=.2)
        local=threading.local(); delayed=set()
        original_append=journal.Journal.append; original_fsync=os.fsync
        def append(j,event,**kwargs):
            append_threads.append(threading.get_ident()); local.kind=event['kind']
            try: return original_append(j,event,**kwargs)
            finally: local.kind=None
        def fsync(fd):
            kind=getattr(local,'kind',None)
            if kind in ('QUIESCE','CANCEL_INTENT') and kind not in delayed:
                delayed.add(kind); start=time.monotonic(); time.sleep(.35)
                slow_spans.append((kind,start,time.monotonic()))
            return original_fsync(fd)
        with patch.object(journal.Journal,'append',append), patch.object(os,'fsync',fsync):
            result=self.run_owner(adapter,sched)
        diagnostic=repr({'result':result,'calls':sched.calls,'times':times,'slow_spans':slow_spans,
                         'callback_threads':callback_threads,'append_threads':append_threads})
        self.assertEqual(result['reason'],'resource gate: quota or role cap exceeded',diagnostic)
        self.assertTrue(result['drained'],diagnostic)
        self.assertEqual([c for c in sched.calls if c[0] in ('submit','cancel')],[('submit',3),('cancel','125')],diagnostic)
        self.assertEqual(delayed,{'QUIESCE','CANCEL_INTENT'},diagnostic)
        self.assertLess(max(y-x for x,y in zip(times,times[1:])),.25,diagnostic)
        for kind,start,end in slow_spans:
            self.assertGreaterEqual(sum(start<t<end for t in times),2,diagnostic)
        self.assertEqual(set(append_threads),{threading.get_ident()},diagnostic)
        self.assertEqual(concurrency[1],1,diagnostic)
        self.assertEqual(len(set(callback_threads)),1,diagnostic)
        self.assertNotIn(threading.get_ident(),callback_threads,diagnostic)
        count=len(times); time.sleep(.12)
        self.assertEqual(len(times),count,'resource observer survived run() return')

    def test_guardian_stop_during_submit_intent_fsync_prevents_issue(self):
        adapter=FakeAdapter(); sched=FakeScheduler(); local=threading.local()
        original_append=journal.Journal.append; original_fsync=os.fsync
        def append(j,event,**kwargs):
            local.kind=event['kind']
            try: return original_append(j,event,**kwargs)
            finally: local.kind=None
        def fsync(fd):
            if getattr(local,'kind',None)=='SUBMIT_INTENT':
                adapter.bad=True; time.sleep(.35)
            return original_fsync(fd)
        with patch.object(journal.Journal,'append',append), patch.object(os,'fsync',fsync):
            result=self.run_owner(adapter,sched)
        rows=journal.read_chain(self.root/'journal.jsonl')
        diagnostic=repr({'result':result,'calls':sched.calls})
        self.assertEqual(result['reason'],'resource gate: quota or role cap exceeded',diagnostic)
        self.assertEqual([c for c in sched.calls if c[0] in ('submit','cancel')],[],diagnostic)
        self.assertEqual(sum(r['event']['kind']=='SUBMIT_INTENT' for r in rows),1,diagnostic)
        self.assertFalse(any(t.name=='evaluation005-resource-guardian' for t in threading.enumerate()))

    def test_guardian_stop_during_complete_fsync_cannot_return_complete(self):
        adapter=FakeAdapter(); local=threading.local()
        original_append=journal.Journal.append; original_fsync=os.fsync
        def append(j,event,**kwargs):
            local.kind=event['kind']
            try: return original_append(j,event,**kwargs)
            finally: local.kind=None
        def fsync(fd):
            if getattr(local,'kind',None)=='COMPLETE':
                adapter.bad=True; time.sleep(.35)
            return original_fsync(fd)
        with patch.object(journal.Journal,'append',append), patch.object(os,'fsync',fsync):
            result=self.run_owner(adapter)
        rows=journal.read_chain(self.root/'journal.jsonl')
        self.assertEqual(result['status'],'HOLD',repr(result))
        self.assertEqual(result['reason'],'resource gate: quota or role cap exceeded',repr(result))
        self.assertEqual(rows[-1]['event']['kind'],'HOLD')
        self.assertFalse(any(t.name=='evaluation005-resource-guardian' for t in threading.enumerate()))

    def test_guardian_is_joined_when_journal_creation_fails(self):
        with patch.object(journal.Journal,'__init__',side_effect=OSError('journal unavailable')):
            with self.assertRaises(OSError): self.run_owner()
        self.assertFalse(any(t.name=='evaluation005-resource-guardian' for t in threading.enumerate()))

    def test_final_guardian_observation_gates_success_before_join(self):
        adapter=FakeAdapter(); original=coordinator.Owner.execute
        def execute(owner):
            result=original(owner)
            adapter.bad=True
            return result
        with patch.object(coordinator.Owner,'execute',execute):
            result=self.run_owner(adapter)
        rows=journal.read_chain(self.root/'journal.jsonl')
        self.assertEqual(result['status'],'HOLD',repr(result))
        self.assertEqual(result['reason'],'resource gate: quota or role cap exceeded',repr(result))
        self.assertEqual(rows[-1]['event']['kind'],'HOLD')
        self.assertFalse(any(t.name=='evaluation005-resource-guardian' for t in threading.enumerate()))

if __name__=='__main__': unittest.main()
