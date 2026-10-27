"""Single writer, one-slot durable controller and closed phase-adapter protocol.

No imports of task/model environments. Task3 must wire a *literal* pinned adapter
in the isolated entry; configuration never names code or executable commands.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import queue
import signal
import threading
import time

from journal import Hold, Journal, acquire_new_lease, decode, exclusive_json, read_chain, read_file, sync_directory
from scheduler import INDICES, TERMINAL, Unknown, IssuePermit

ISSUE_SECONDS=3630
ACTIVE_SECONDS=1530
PENDING_SECONDS=900
POLICY_SECONDS=1200
POST_SECONDS=1200
DRAIN_SECONDS=180
FINAL_SECONDS=120
LAG_SECONDS=30
CONTROLLER_SECONDS=28800
LIVE_INTERVAL=.1   # leaves .15s for bounded resource observation
SCHEDULER_INTERVAL=4  # plus <=10.5s bounded complete scheduler observation

@dataclass(frozen=True)
class Runtime:
    root: Path
    context: dict

class PhaseAdapter:
    """Closed protocol; methods raise on failure/unknown, never submit/cancel.

    All methods receive context (root-authorized nonce/source/config/controller,
    runtime deadline and stop Event). Non-resource operations are bounded by the
    context deadline; resource_check must complete within .1s, or its latency is
    a HOLD. It authenticates all18 reservation/current quota and separately shared
    source/controls/derived usage, never raw/free-filesystem credit. reserve returns
    exactly descriptor/ledger/reservation JSON; the owner exclusively fsyncs it.
    terminal preserves *every* terminal outcome, independently verifies process
    exit+EOF and reviews/seals; returns qualified=True only for a proven success.
    It must honor quiescence and cannot publish after its context deadline.
    """
    def prepare_once(self,phase,context): raise NotImplementedError
    def preflight_current(self,phase,index,context): raise NotImplementedError
    def reserve(self,phase,index,context): raise NotImplementedError
    def resource_check(self,transaction,context): raise NotImplementedError
    def terminal(self,phase,index,transaction,identity,observation,context): raise NotImplementedError
    def phase_complete(self,phase,context): raise NotImplementedError

class Worker:
    """One bounded operation, result only; no journal or lease mutations."""
    def __init__(self,operation):
        self.result=queue.Queue(maxsize=1)
        def invoke():
            try: self.result.put((True,operation()))
            except BaseException as exc: self.result.put((False,exc))
        self.thread=threading.Thread(target=invoke,daemon=True)
        self.thread.start()

class Owner:
    def __init__(self,runtime,adapter,scheduler,stop,clock,controller):
        self.root=Path(runtime.root); self.adapter=adapter; self.scheduler=scheduler
        self.clock=clock; self.stop=stop
        remaining=min(CONTROLLER_SECONDS,controller['end_epoch']-time.time())
        self.deadline=clock()+remaining
        self.context=dict(runtime.context,root=str(self.root),deadline=self.deadline,stop=stop)
        self.journal=Journal(self.root/'journal.jsonl')
        self.transaction=None; self.identity=None; self.submission=None
        self.halt=None; self.halt_observed_at=None; self.cancelled=False; self.drained=False
        self.pending_started=None
        self.journal.append({'kind':'ACTIVATE','context':runtime.context,'controller':controller,
                             'requested_resources':{'seconds':28800,'cpus':1,'memory_mib':2048}})
    def record(self,kind,**data): return self.journal.append(dict(data,kind=kind),emergency=self.halt is not None)
    def halt_once(self,reason):
        if self.halt is None:
            self.halt=str(reason)[:2048]; self.halt_observed_at=time.monotonic(); self.stop.set()
            self.record('QUIESCE',reason=self.halt)
    def live(self):
        if self.stop.is_set(): self.halt_once('shutdown signal')
        if self.submission and self.deadline-self.clock()<ACTIVE_SECONDS:
            self.halt_once('active remaining lifetime below1530')
        if self.pending_started is not None and self.clock()-self.pending_started>PENDING_SECONDS:
            self.halt_once('pending exceeded900')
        started=time.monotonic()
        try:
            result=self.adapter.resource_check(self.transaction,self.context)
            if not isinstance(result,dict): raise Hold('unknown resource observation')
        except BaseException as exc: self.halt_once('resource gate: '+str(exc))
        if time.monotonic()-started>.1: self.halt_once('resource observation exceeded .1s')
    def wait(self,operation,timeout,*,allow_halt=False):
        worker=Worker(operation); end=time.monotonic()+timeout
        while True:
            self.live()
            try:
                ok,value=worker.result.get(timeout=min(LIVE_INTERVAL,max(.001,end-time.monotonic())))
                if not ok:
                    if isinstance(value,Unknown): self.record('RPC_UNKNOWN',raw=value.raw,reason=str(value))
                    raise value
                if self.halt and not allow_halt: raise Hold(self.halt)
                return value
            except queue.Empty:
                if time.monotonic()>=end: raise Hold('bounded operation timed out; outcome unknown')
    def gate(self,operation,timeout=POST_SECONDS,*,reservation=False):
        self.live()
        if self.halt: raise Hold(self.halt)
        context=dict(self.context,deadline=min(self.deadline-FINAL_SECONDS,self.clock()+timeout))
        value=self.wait(lambda:operation(context),min(timeout,max(.001,self.deadline-self.clock()-FINAL_SECONDS)))
        if value is False or (not reservation and value is not None): raise Hold('adapter returned failed/unknown gate')
        return value
    def observe(self):
        def snapshot():
            controller=self.scheduler.controller()
            observation=self.scheduler.observe(self.submission)
            return controller,observation
        controller,observation=self.wait(snapshot,10.5,allow_halt=True)
        if controller['job_id']!=self.context['controller_job_id'] or controller['user']!=self.context['uid'] or controller['state']!='RUNNING':
            raise Hold('controller identity changed')
        self.deadline=min(self.deadline,self.clock()+controller['end_epoch']-time.time())
        self.context['deadline']=self.deadline
        identity=observation['identity']
        if (identity.get('array_job_id')!=self.submission['job_id'] or
            identity.get('index')!=self.submission['index'] or
            identity.get('name')!=self.submission['name'] or identity.get('user')!=self.context['uid']):
            raise Hold('unowned materialized identity')
        if self.identity and self.identity!=identity: raise Hold('materialized identity changed')
        self.identity=identity
        self.record('CONTROLLER_OBSERVATION',controller=controller)
        self.record('OBSERVATION',observation=observation)
        state=observation['state']
        if state=='PENDING':
            if self.pending_started is None: self.pending_started=self.clock()
        else: self.pending_started=None
        if observation['terminal'] is not (state in TERMINAL): raise Hold('contradictory terminal observation')
        return observation
    def drain(self,observation=None):
        self.halt_once(self.halt or 'drain requested')
        if self.identity is None and self.submission is not None:
            try: observation=self.observe()
            except BaseException as exc:
                self.record('DRAIN_UNKNOWN',reason=str(exc)); return None
        if self.identity is None: return observation
        if observation and observation['terminal']:
            self.drained=True; return observation
        if not self.cancelled:
            self.cancelled=True
            self.record('CANCEL_INTENT',identity=self.identity)
            try:
                raw=self.wait(lambda:self.scheduler.cancel(self.identity),10.5,allow_halt=True)
                self.record('CANCEL_RESULT',raw=raw)
                if raw.get('returncode')!=0 or not raw.get('complete_eof'): return None
            except BaseException as exc:
                self.record('CANCEL_UNKNOWN',reason=str(exc)); return None
        end=min(self.deadline-self.clock()-FINAL_SECONDS,DRAIN_SECONDS)
        drain_deadline=time.monotonic()+max(0,end)
        while time.monotonic()<drain_deadline:
            try: observation=self.observe()
            except BaseException as exc:
                self.record('DRAIN_UNKNOWN',reason=str(exc)); return None
            if observation['terminal']:
                self.drained=True; return observation
            self.pause(min(SCHEDULER_INTERVAL,max(0,drain_deadline-time.monotonic())))
        self.record('DRAIN_UNKNOWN',reason='terminal drain deadline'); return None
    def pause(self,seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            self.live()
            if self.halt: return
            self.stop.wait(min(LIVE_INTERVAL,max(0,end-time.monotonic())))
    def preserve(self,phase,index,observation):
        # This includes failed/cancelled outcomes. The adapter cannot qualify them.
        context=dict(self.context,deadline=min(self.deadline-FINAL_SECONDS,self.clock()+POST_SECONDS))
        result=self.wait(lambda:self.adapter.terminal(phase,index,self.transaction,self.identity,observation,context),
                         max(.001,min(POST_SECONDS,self.deadline-self.clock()-FINAL_SECONDS)),allow_halt=True)
        self.record('TERMINAL_PROCESSED',index=index,result=result)
        if observation['state']!='COMPLETED' or observation.get('exit_code','0:0')!='0:0' or not isinstance(result,dict) or result.get('qualified') is not True:
            raise Hold('terminal technical gate failed')
    def slot(self,phase,index):
        self.transaction=None; self.identity=None; self.submission=None
        self.cancelled=False; self.pending_started=None; self.drained=False
        self.live()
        if self.halt or self.deadline-self.clock()<ISSUE_SECONDS: raise Hold(self.halt or 'issue lifetime below3630')
        self.gate(lambda context:self.adapter.preflight_current(phase,index,context))
        transaction=self.gate(lambda context:self.adapter.reserve(phase,index,context),reservation=True)
        if not isinstance(transaction,dict) or set(transaction)!={'descriptor','ledger','reservation'}: raise Hold('adapter reservation schema')
        if transaction['descriptor'].get('index')!=index or transaction['descriptor'].get('nonce')!=self.context['nonce'] or transaction['reservation'].get('index')!=index:
            raise Hold('reservation index/nonce binding')
        references={}
        for role in ('descriptor','ledger','reservation'):
            references[role+'_reference']=exclusive_json(self.root/('slot%02d-%s.json'%(index,role)),transaction[role])
        reference=references['reservation_reference']
        self.transaction=dict(transaction,**references)
        self.record('RESERVATION',index=index,reference=reference)
        name='z1a005-'+self.context['nonce']+'-%02d'%index
        log=Path(self.context['log_root'])/('slot%02d'%index)/'slurm'
        log.mkdir(mode=0o700,exist_ok=False)  # Phase adapter prepared exact provenance parent.
        sync_directory(log.parent); sync_directory(log)
        # Last live/deadline/shutdown gate immediately before durable intent.
        self.live()
        if self.halt or self.deadline-self.clock()<ISSUE_SECONDS: raise Hold(self.halt or 'issue lifetime below3630')
        self.pending_started=self.clock()
        self.record('SUBMIT_INTENT',phase=phase,index=index,name=name,log=str(log),reservation=reference)
        # Once intent exists, ANY ambiguous result consumes this slot forever.
        permit=IssuePermit(self.stop,time.monotonic()+2)
        self.submission=self.wait(lambda:self.scheduler.submit(phase,index,name,str(log),permit=permit),10.5,allow_halt=True)
        observation=None
        try:
            self.record('SUBMIT_RESULT',submission=self.submission)
            while True:
                observation=self.observe()
                self.live()
                if observation['terminal']:
                    self.drained=True; break
                if self.halt:
                    observation=self.drain(observation); break
                self.pause(SCHEDULER_INTERVAL)
        except BaseException as exc:
            self.halt_once(exc)
            observation=self.drain(observation)
        if observation and observation['terminal']: self.preserve(phase,index,observation)
        else: raise Hold('terminal outcome unknown; preserve partials and reconcile only')
        if self.halt: raise Hold(self.halt)
        self.record('SLOT_COMPLETE',index=index)
    def execute(self):
        try:
            for phase,indices in INDICES.items():
                self.gate(lambda context:self.adapter.prepare_once(phase,context))
                self.record('PHASE_PREPARED',phase=phase)
                for index in indices: self.slot(phase,index)
                self.gate(lambda context:self.adapter.phase_complete(phase,context))
                self.record('PHASE_COMPLETE',phase=phase)
            self.record('COMPLETE'); status='COMPLETE'
        except BaseException as exc:
            self.halt_once(exc)
            self.record('HOLD',reason=self.halt,identity=self.identity,submission=self.submission)
            status='HOLD'
        return {'status':status,'reason':self.halt,'drained':self.drained,'halt_observed_at':self.halt_observed_at}

def run(config,adapter,scheduler,*,stop=None,clock=time.monotonic):
    """Internal reviewed-code API; isolated entry authenticates Runtime first.

    Nested capability and genuine controller identity precede permanent ownership.
    No cleanup removes the lease/reservations/partial journal. A second call fails.
    """
    if os.path.lexists(Path(config.root)/'owner.lease') or os.path.lexists(Path(config.root)/'journal.jsonl'):
        raise Hold('existing permanent state; reconcile only before scheduler access')
    stop=stop or threading.Event()
    context=config.context
    controller=scheduler.controller()
    if controller['job_id']!=context['controller_job_id'] or controller['user']!=context['uid'] or controller['state']!='RUNNING':
        raise Hold('controller identity mismatch')
    if controller['end_epoch']<=time.time(): raise Hold('expired controller')
    # Both exact phase batch requests must pass nested compute capability.
    capabilities=[]
    for phase,indices in INDICES.items():
        raw=scheduler.capability(phase,indices[0],'z1a005-'+context['nonce']+'-%02d'%indices[0],str(config.root))
        if raw.get('returncode')!=0 or not raw.get('complete_eof'): raise Hold('nested sbatch capability failed/unknown')
        capabilities.append({'phase':phase,'raw':raw})
    controller=dict(controller,nested_capability=capabilities)
    lease=acquire_new_lease(config.root,context)
    previous={}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
            previous[sig]=signal.signal(sig,lambda *_:stop.set())
    try: return Owner(config,adapter,scheduler,stop,clock,controller).execute()
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)
        lease.close()

def reconcile(config):
    root=Path(config.root if isinstance(config,Runtime) else config)
    chain=read_chain(root/'journal.jsonl')
    lease=decode(read_file(root/'owner.lease',16384)) if (root/'owner.lease').exists() else None
    return {'status':'READ_ONLY','lease':lease,'journal':chain,'can_submit':False}
