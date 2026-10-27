"""Fixed Slurm operations and bounded child transport; deliberately no retry API."""
import base64
from dataclasses import dataclass
import datetime
import hashlib
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import threading
import time

from journal import Hold, read_file

CPU_ROOT='/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002'
CONTROL_ROOT=CPU_ROOT+'/admission-staged-integration002/autonomous005'
INDICES={'remaining':(3,6,9,12,15),'science':(1,2,4,5,7,8,10,11,13,14,16,17)}
BATCHES={phase:CONTROL_ROOT+'/controller-inputs/'+phase+'.sbatch' for phase in INDICES}
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED'}

@dataclass(frozen=True)
class IssuePermit:
    stop: threading.Event
    deadline: float
    def valid(self): return not self.stop.is_set() and time.monotonic()<self.deadline

class Unknown(Hold):
    def __init__(self,message,raw): super().__init__(message); self.raw=raw

def child_environment(environment=None):
    env=dict(os.environ if environment is None else environment)
    return {k:v for k,v in env.items() if not k.startswith(('SLURM_','SBATCH_')) and k not in {'PYTHONPATH','PYTHONHOME'}}

def run_process(argv, *, timeout, cap=131072, environment=None, permit=None):
    """Reap plus EOF, with output/elapsed bounds even when descendants hold pipes."""
    started=time.monotonic(); deadline=started+timeout
    result={'argv':list(argv),'returncode':None,'stdout':'','stderr':'','complete_eof':False,
            'outcome':'UNKNOWN','pid':None,'observed_ppid':os.getpid()}
    if permit is not None and (not isinstance(permit,IssuePermit) or not permit.valid()):
        result['outcome']='NOT_STARTED'; result['stderr']='revoked/expired issue permit'; return result
    try:
        p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                           env=child_environment(environment),start_new_session=True,close_fds=True)
    except OSError as exc:
        result['stderr']=str(exc); return result
    result['pid']=p.pid
    buffers={'stdout':bytearray(),'stderr':bytearray()}; counts={'stdout':0,'stderr':0}; selector=selectors.DefaultSelector()
    for key,pipe in [('stdout',p.stdout),('stderr',p.stderr)]:
        os.set_blocking(pipe.fileno(),False); selector.register(pipe,selectors.EVENT_READ,key)
    failed=False
    try:
        while selector.get_map() or p.poll() is None:
            if time.monotonic()>=deadline:
                failed=True; break
            for key,_ in selector.select(min(.05,max(0,deadline-time.monotonic()))):
                block=os.read(key.fd,65536)
                if not block:
                    selector.unregister(key.fileobj); key.fileobj.close(); continue
                counts[key.data]+=len(block)
                buf=buffers[key.data]
                room=cap-len(buf); buf.extend(block[:room])
                if len(block)>room: failed=True; break
            if failed: break
        if failed:
            try: os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError: pass
        try: result['returncode']=p.wait(timeout=1)
        except subprocess.TimeoutExpired: failed=True
        result['complete_eof']=not selector.get_map() and not failed
        if result['complete_eof']: result['outcome']='KNOWN'
    finally:
        for key in list(selector.get_map().values()): key.fileobj.close()
        selector.close()
    for key in buffers:
        result[key]=buffers[key].decode('utf-8',errors='replace')
        result[key+'_base64']=base64.b64encode(buffers[key]).decode('ascii')
        result[key+'_bytes']=counts[key]
        result[key+'_sha256']=hashlib.sha256(buffers[key]).hexdigest()
        result[key+'_truncated']=counts[key]>len(buffers[key])
    result['elapsed_seconds']=time.monotonic()-started
    return result

def policy_argv(phase,index,name,log,*,test=False):
    if phase not in INDICES or type(index) is not int or index not in INDICES[phase]: raise Hold('forbidden index')
    if not re.fullmatch(r'z1a005-[0-9a-f]{32,64}-[0-9]{2}',name): raise Hold('job name')
    if not Path(log).is_absolute() or '\n' in log: raise Hold('log path')
    argv=['/usr/bin/sbatch','--parsable', '--array='+str(index)+'%1','--job-name='+name,
          '--output='+log+'/%A_%a.log','--error='+log+'/%A_%a.err']
    if test: argv.append('--test-only')
    return argv+[BATCHES[phase]]

def fields(control):
    return dict(re.findall(r'(\w+)=([^\s]+)',control))

def accounting_rows(text):
    rows=[]
    for line in text.splitlines():
        parts=line.split('|')
        if len(parts)==8 and re.fullmatch(r'[1-9][0-9]*',parts[0]): rows.append(parts)
    return rows

def materialize(submission,control,accounting,uid):
    values=fields(control)
    rows=[r for r in accounting_rows(accounting) if r[0]==values.get('JobId')]
    if len(rows)!=1: raise Hold('missing/ambiguous materialized accounting')
    row=rows[0]
    user=re.fullmatch(r'[^()]+\(([0-9]+)\)',values.get('UserId',''))
    if (not user or user[1]!=uid or row[2]!=uid or
        values.get('JobName')!=submission['name'] or row[1]!=submission['name'] or
        values.get('ArrayJobId')!=submission['job_id'] or
        values.get('ArrayTaskId')!=str(submission['index'])):
        raise Hold('scheduler identity mismatch')
    cstate=values.get('JobState'); astate=row[3].split()[0].rstrip('+')
    if cstate!=astate or cstate not in TERMINAL|ACTIVE: raise Hold('ambiguous scheduler state')
    return {'job_id':row[0], 'array_job_id':values['ArrayJobId'], 'index':submission['index'],
            'name':row[1], 'user':uid}

class Scheduler:
    def __init__(self,controller_job_id,uid,batches):
        if not re.fullmatch(r'[1-9][0-9]*',controller_job_id): raise Hold('controller job id')
        self.controller_job_id=controller_job_id; self.uid=uid
        if not isinstance(batches,dict) or set(batches)!=set(BATCHES): raise Hold('closed batch inventory')
        for phase,reference in batches.items():
            if set(reference)!={'path','sha256'} or reference['path']!=BATCHES[phase] or not re.fullmatch('[0-9a-f]{64}',reference['sha256']):
                raise Hold('batch authority binding')
        self.batches=batches
    def _bound_batch(self,phase):
        reference=self.batches[phase]
        raw=read_file(reference['path'])
        if hashlib.sha256(raw).hexdigest()!=reference['sha256']: raise Hold('batch source changed')
    def _rpc(self,argv,timeout=2,*,permit=None):
        raw=run_process(argv,timeout=timeout,cap=32768,permit=permit)
        if raw['outcome']!='KNOWN' or raw['returncode']!=0: raise Unknown('scheduler command failed/unknown',raw)
        return raw
    def capability(self,phase,index,name,log):
        self._bound_batch(phase)
        return self._rpc(policy_argv(phase,index,name,log,test=True))
    def controller(self):
        raw=self._rpc(['/usr/bin/scontrol','show','job','--oneliner',self.controller_job_id])
        f=fields(raw['stdout']); user=re.fullmatch(r'[^()]+\(([0-9]+)\)',f.get('UserId',''))
        try: end=datetime.datetime.fromisoformat(f['EndTime']).timestamp()
        except (ValueError,KeyError) as exc: raise Unknown('unknown controller end time',raw) from exc
        if f.get('JobId')!=self.controller_job_id or not user or user[1]!=self.uid or f.get('JobState')!='RUNNING':
            raise Unknown('controller identity/state',raw)
        return {'job_id':self.controller_job_id,'user':self.uid,'state':f['JobState'],'end_epoch':end,
                'cpus':int(f['NumCPUs']),'memory':f.get('MinMemoryNode',f.get('MinMemoryCPU','unknown')),'raw':raw}
    def submit(self,phase,index,name,log,*,permit):
        self._bound_batch(phase)
        if not isinstance(permit,IssuePermit): raise Hold('issue permit required')
        raw=self._rpc(policy_argv(phase,index,name,log),permit=permit)
        match=re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_-]+)?\n?',raw['stdout'])
        if not match: raise Unknown('unknown sbatch result',raw)
        return {'job_id':match[1],'index':index,'name':name,'raw':raw}
    def observe(self,submission):
        acct=self._rpc(['/usr/bin/sacct','--noheader','--parsable2','--allocations','--jobs='+submission['job_id'],
                        '--format=JobIDRaw,JobName%128,UID,State,Submit,Start,End,ExitCode'])
        candidates=[r for r in accounting_rows(acct['stdout']) if r[1]==submission['name'] and r[2]==self.uid]
        # Job IDs come ONLY from actual accounting records, never array arithmetic.
        identities=[]; controls=[]
        if len(candidates)>2: raise Unknown('ambiguous accounting inventory',acct)
        for row in candidates:
            raw=self._rpc(['/usr/bin/scontrol','show','job','--oneliner',row[0]])
            controls.append(raw)
            try: identity=materialize(submission,raw['stdout'],acct['stdout'],self.uid)
            except Hold: continue
            identities.append((identity,fields(raw['stdout'])['JobState'],row))
        if len(identities)!=1: raise Unknown('empty/ambiguous materialized identity',{'accounting':acct,'controls':controls})
        identity,state,row=identities[0]
        return {'identity':identity,'state':state,'terminal':state in TERMINAL,'exit_code':row[7],
                'raw':[acct]+controls}
    def cancel(self,identity):
        if identity.get('user')!=self.uid or not re.fullmatch(r'[1-9][0-9]*',identity.get('job_id','')):
            raise Hold('unowned cancellation')
        return self._rpc(['/usr/bin/scancel',identity['job_id']])
