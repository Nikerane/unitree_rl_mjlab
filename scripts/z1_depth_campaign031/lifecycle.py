"""Finite031 lifecycle; audited020 pidfd ownership/scanner retained below.

The process/scanner section changes only its strict-helper import.
"""
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from . import files as a


def process_identity(pid):
    """Linux kernel startticks distinguish PID reuse; no portable fallback."""
    raw=Path(f'/proc/{pid}/stat').read_text()
    # comm may contain spaces/parentheses; fields after final ')' start at3.
    ticks=raw.rsplit(')',1)[1].split()[19]
    a.need(ticks.isdigit(),'invalid process startticks')
    return dict(pid=pid,startticks=ticks)


class _LinuxProcesses:
    """Dedicated Linux subreaper backend; capability work is lazy and fail closed."""
    def __init__(self): self.parent_pid=os.getpid();self.parent_start=None
    def _row(self,pid):
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        a.need(len(fields)>=20,'ownership uncertainty: malformed /proc identity')
        return dict(pid=pid,state=fields[0],ppid=int(fields[1]),pgid=int(fields[2]),sid=int(fields[3]),startticks=int(fields[19]))
    def _children(self,pid):
        """Read all task children; missing API is distinct from an exit race."""
        task=Path('/proc')/str(pid)/'task'
        threads=[]
        try:
            for thread in task.iterdir():
                a.need(len(threads)<65536,'ownership uncertainty: thread snapshot cap exceeded')
                threads.append(thread)
        except FileNotFoundError:
            try:self._row(pid)
            except FileNotFoundError:
                a.need(pid!=self.parent_pid,'ownership uncertainty: supervisor disappeared')
                return ()
            raise ValueError('ownership uncertainty: required children API missing')
        a.need(len(threads)<=65536,'ownership uncertainty: thread snapshot cap exceeded')
        children=set()
        for thread in threads:
            if not thread.name.isdigit():continue
            try:
                with (thread/'children').open('r',encoding='ascii') as stream:body=stream.read(2*1024**2+1)
            except FileNotFoundError:
                try:thread.stat()
                except FileNotFoundError:continue  # task exited during scan
                raise ValueError('ownership uncertainty: required children API missing')
            a.need(len(body)<=2*1024**2,'ownership uncertainty: children API byte cap exceeded')
            for token in body.split():
                a.need(token.isdigit() and int(token)>0,'ownership uncertainty: malformed child PID')
                children.add(int(token))
                a.need(len(children)<=65536,'ownership uncertainty: child snapshot cap exceeded')
        return children
    def snapshot(self):
        # Rooted at this dedicated subreaper, never enumerate foreign /proc/PID
        # entries. A child forked by any thread is included via task/*/children.
        rows={};seen=set();pending=[self.parent_pid];queued={self.parent_pid}
        while pending:
            pid=pending.pop();queued.remove(pid)
            if pid in seen:continue
            a.need(len(seen)<65536,'ownership uncertainty: process snapshot cap exceeded')
            seen.add(pid)
            try:row=self._row(pid)
            except FileNotFoundError:
                a.need(pid!=self.parent_pid,'ownership uncertainty: supervisor disappeared')
                continue
            children=self._children(pid)
            try:after=self._row(pid)
            except FileNotFoundError:
                a.need(pid!=self.parent_pid,'ownership uncertainty: supervisor disappeared')
                continue
            a.need(after['startticks']==row['startticks'],'ownership uncertainty: process identity changed during traversal')
            rows[pid]=after
            for child in children:
                if child in seen or child in queued:continue
                a.need(len(seen)+len(queued)<65536,'ownership uncertainty: traversal queue cap exceeded')
                pending.append(child);queued.add(child)
        parent=rows.get(self.parent_pid)
        a.need(parent is not None and (self.parent_start is None or parent['startticks']==self.parent_start),'ownership uncertainty: supervisor identity changed')
        return rows
    def require_empty(self):
        rows=self.snapshot()
        a.need(not any(r['ppid']==self.parent_pid for r in rows.values()),'ownership uncertainty: pre-existing direct children')
    def prepare(self):
        import ctypes
        import platform
        a.need(sys.platform=='linux' and platform.machine()=='x86_64','qualified Linux x86_64 ownership backend required')
        a.need(all(hasattr(os,k) for k in ('waitid','P_ALL','WEXITED','WNOHANG','WNOWAIT')),'nonreaping child proof unavailable')
        self.require_empty();self.parent_start=self._row(self.parent_pid)['startticks']
        libc=ctypes.CDLL(None,use_errno=True);self._ctypes=ctypes;self._libc=libc
        # libc exports when available, otherwise exact x86_64 kernel ABI only.
        if hasattr(libc,'pidfd_open') and hasattr(libc,'pidfd_send_signal'):
            op=libc.pidfd_open;op.argtypes=(ctypes.c_int,ctypes.c_uint);op.restype=ctypes.c_int
            send=libc.pidfd_send_signal;send.argtypes=(ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint);send.restype=ctypes.c_int
            self._open=lambda pid:op(pid,0)
            self._send=lambda fd,sig:send(fd,sig,None,0)
        else:
            syscall=libc.syscall;syscall.restype=ctypes.c_long
            self._open=lambda pid:syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
            self._send=lambda fd,sig:syscall(ctypes.c_long(424),ctypes.c_int(fd),ctypes.c_int(sig),ctypes.c_void_p(),ctypes.c_uint(0))
        # No child is constructed unless subreaper, pidfd and waitid capability
        # are proved in this already-counted software/setup preflight.
        a.need(libc.prctl(36,1,0,0,0)==0,'PR_SET_CHILD_SUBREAPER unsupported')
        enabled=ctypes.c_int()
        a.need(libc.prctl(37,ctypes.byref(enabled),0,0,0)==0 and enabled.value==1,'subreaper activation not verified')
        fd=self._open(self.parent_pid)
        a.need(fd>=0,'pidfd_open unsupported')
        try:a.need(self._send(fd,0)==0,'pidfd_send_signal unsupported')
        finally:os.close(fd)
        a.need(self.no_children(),'unexpected direct child before issue');self.require_empty()
    def open_identity(self,identity):
        import errno
        fd=self._open(identity[0])
        if fd<0:
            if self._ctypes.get_errno()==errno.ESRCH:return None
            raise ValueError('ownership uncertainty: pidfd_open failed')
        try:
            row=self._row(identity[0])
            if row['startticks']!=identity[1]:os.close(fd);return None
        except FileNotFoundError:os.close(fd);return None
        except BaseException:os.close(fd);raise
        return fd
    def signal(self,fd,sig):
        import errno
        if self._send(fd,sig)!=0:
            a.need(self._ctypes.get_errno()==errno.ESRCH,'ownership uncertainty: pidfd signal failed')
    def reap(self,identities,leader_pid):
        for pid,ticks in identities:
            if pid==leader_pid:continue  # Popen alone owns leader wait/status.
            try:row=self._row(pid)
            except FileNotFoundError:continue
            if row['startticks']!=ticks or row['ppid']!=self.parent_pid or row['state']!='Z':continue
            try:reaped,_=os.waitpid(pid,os.WNOHANG)
            except ChildProcessError:continue
            a.need(reaped in (0,pid),'ownership uncertainty: wrong adopted child reaped')
    def no_children(self):
        try:os.waitid(os.P_ALL,0,os.WEXITED|os.WNOHANG|os.WNOWAIT)
        except ChildProcessError:return True
        return False  # includes a live child or an unreaped exit; never reap here
    def close(self,fd):os.close(fd)


class OwnedTree:
    """Track descendants and adopted orphans by PID/startticks, never PGID/SID.

    Requires a dedicated subreaper parent with no unrelated direct children.
    Snapshot/open/signal/reap are injected for pure independent kernel fixtures.
    """
    def __init__(self,api,leader_pid):
        self.api=api;self.leader_pid=leader_pid;self.identities={};self.rows={}
    def refresh(self):
        try:
            rows=self.api.snapshot();a.need(type(rows) is dict and len(rows)<=65536,'process snapshot cap/type differs')
            for pid,row in rows.items():
                a.need(type(pid) is int and pid>0 and type(row) is dict and set(row)=={'pid','ppid','pgid','sid','state','startticks'},'process identity fields differ')
                a.need(row['pid']==pid and all(type(row[k]) is int and row[k]>=0 for k in ('pid','ppid','pgid','sid','startticks')) and type(row['state']) is str and len(row['state'])==1,'process identity values differ')
            owned={pid for pid,r in rows.items() if r['ppid']==self.api.parent_pid or (pid,r['startticks']) in self.identities}
            while True:
                expanded=owned|{pid for pid,r in rows.items() if r['ppid'] in owned}
                if expanded==owned:break
                owned=expanded
                a.need(len(owned)<=4096,'owned descendant cap exceeded')
            for pid in owned:
                identity=(pid,rows[pid]['startticks'])
                if identity not in self.identities:
                    a.need(len(self.identities)<4096,'owned identity cap exceeded')
                    self.identities[identity]=self.api.open_identity(identity)
            self.rows={pid:rows[pid] for pid in owned}
            return self.rows
        except BaseException as exc:
            raise ValueError('ownership uncertainty: snapshot/identity tracking failed') from exc
    def signal_all(self,sig):
        error=None
        try:self.refresh()
        except BaseException as exc:error=exc
        # On an uncertain scan still stop every identity already pinned; never
        # turn a partial scan into clearance or signal an unpinned numeric PID.
        for identity,fd in self.identities.items():
            if fd is None:continue
            try:self.api.signal(fd,sig)
            except BaseException as exc:error=error or exc
        if error is not None:raise ValueError('ownership uncertainty: owned signal/refresh failed') from error
    def pending(self):
        self.refresh();self.api.reap(tuple(self.identities),self.leader_pid);self.refresh()
        live=any(row['state'] not in ('Z','X') and (row['pid'],row['startticks']) in self.identities for row in self.rows.values())
        return live or not self.api.no_children()
    def close(self):
        for fd in self.identities.values():
            if fd is not None:self.api.close(fd)
        self.identities.clear()


class _Child:
    def __init__(self,p,api):self.p=p;self.owned=OwnedTree(api,p.pid)
    def poll(self):self.owned.refresh();return self.p.poll()
    def terminate_group(self):self.owned.signal_all(signal.SIGTERM)
    def kill_group(self):self.owned.signal_all(signal.SIGKILL)
    def group_alive(self):return self.owned.pending()
    def wait(self,timeout=None):return self.p.wait(timeout=timeout)
    def close_ownership(self):self.owned.close()


class _Runner:
    def __init__(self):self.api=_LinuxProcesses();self.api.prepare()
    def start(self,command,env,stdout,stderr):
        self.api.require_empty()
        # Return the owned handle without any fallible postspawn kernel reads.
        # First refresh happens only once parent has saved it under signal mask.
        return _Child(subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,
            stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True),self.api)


def _cleanup(child,*,clock=time.monotonic,sleep=time.sleep):
    """Abort owned descendants; uncertainty never skips best-effort stop/reap."""
    errors=[]
    def attempt(fn):
        try:return fn()
        except BaseException as exc:errors.append(exc)
    attempt(child.terminate_group)
    try:child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        attempt(child.kill_group);attempt(lambda:child.wait(timeout=5))
    except BaseException as exc:
        errors.append(exc);attempt(child.kill_group)
    else:
        # A completed leader does not establish disappearance of descendants.
        attempt(child.kill_group)
    deadline=clock()+1
    while True:
        try:alive=child.group_alive()
        except BaseException as exc:
            raise ValueError('cleanup uncertainty: owned descendant observation failed') from exc
        a.need(type(alive) is bool,'cleanup uncertainty: invalid owned descendant observation')
        if not alive:break
        a.need(clock()<deadline,'cleanup uncertainty: owned descendants remain')
        attempt(child.kill_group)  # rescan newly forked/adopted identities
        sleep(.01)
    close=getattr(child,'close_ownership',None)
    if close is not None:attempt(close)
    if errors:raise ValueError('cleanup uncertainty: signal/reap operation failed') from errors[0]


def tree_bytes(path):
    """Snapshot regular bytes; only vanished entries are an ignorable race.

    Directory descriptors prevent symlink traversal after discovery. Each entry
    has one non-following stat snapshot for its type and size. A directory open
    is checked against that identity before scanning; replacement fails closed.
    """
    def kind(mode):
        for check,label in ((stat.S_ISREG,'regular'),(stat.S_ISDIR,'directory'),
                            (stat.S_ISLNK,'symlink'),(stat.S_ISFIFO,'fifo'),
                            (stat.S_ISSOCK,'socket'),(stat.S_ISCHR,'character-device'),
                            (stat.S_ISBLK,'block-device')):
            if check(mode): return label
        return 'unknown'

    def snapshot(name,display,dir_fd=None):
        try: return os.stat(name,dir_fd=dir_fd,follow_symlinks=False)
        except FileNotFoundError: return None
        except OSError as exc:
            raise ValueError(f'{display}: type=unreadable; lstat failed: {exc}') from exc

    def directory(name,display,before,dir_fd=None):
        a.need(stat.S_ISDIR(before.st_mode),f'{display}: type={kind(before.st_mode)}; directory required')
        try: fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=dir_fd)
        except FileNotFoundError: return 0
        except OSError as exc:
            raise ValueError(f'{display}: type=directory-open-failure (non-directory/symlink/unreadable); {exc}') from exc
        try:
            opened=os.fstat(fd)
            a.need((before.st_dev,before.st_ino)==(opened.st_dev,opened.st_ino),
                   f'{display}: type={kind(opened.st_mode)}; directory identity changed')
            try:
                with os.scandir(fd) as entries:
                    discovered=[(entry.name,entry.is_dir(follow_symlinks=False)) for entry in entries]
            except FileNotFoundError: return 0
            except OSError as exc:
                raise ValueError(f'{display}: type=directory; listing failed: {exc}') from exc
            total=0
            for child,was_directory in discovered:
                child_path=display/child
                item=snapshot(child,child_path,fd)
                if item is None: continue
                item_kind=kind(item.st_mode)
                a.need(item_kind in ('regular','directory'),f'{child_path}: type={item_kind}; forbidden case/control entry')
                a.need(was_directory==stat.S_ISDIR(item.st_mode),f'{child_path}: type={item_kind}; discovered file type changed')
                if item_kind=='regular': total+=item.st_size
                else: total+=directory(child,child_path,item,fd)
            return total
        finally: os.close(fd)

    path=Path(path)
    root=snapshot(path,path)
    return 0 if root is None else directory(path,path,root)

# New031 finite authority, quota and time domain. No historical authority imports.
import math
import uuid
import re
import socket
from .files import canonical, digest, need, read_json, reference, verify_ref, write_once
from .manifest import ROOT, EVAL_ROOT, CAP_BYTES, GIB, LIMITS


def required_free(used):
    need(type(used) is int and 0<=used<=CAP_BYTES, '031 used bytes outside reserved cap')
    return CAP_BYTES-used+3*GIB


def admit_quota(free, used):
    need(type(free) is int and free>=required_free(used), 'HOLD_UNCLAIMED: insufficient remaining031 reservation plus floor/spare')
    return dict(free_bytes=free,used031_bytes=used,required_free_bytes=required_free(used))


def current_quota():
    """Ceph logical quota, never filesystem-wide free space."""
    root='/cephhome/eunikhilr'
    maximum=int(os.getxattr(root,'ceph.quota.max_bytes'))
    used=int(os.getxattr(root,'ceph.dir.rbytes'))
    consumed=sum(tree_bytes(path) for path in (ROOT,EVAL_ROOT) if Path(path).exists())
    return admit_quota(maximum-used,consumed)


def remote_host(host=None):
    host=socket.gethostname().split('.')[0] if host is None else host
    need(bool(re.fullmatch(r'v[a-z]+[0-9]+',host)), 'Vega/Slurm clock domain required')
    return host


def start_clock(path, manifest_sha, job_id, allocation_start, *, host=None):
    remote_host(host)
    need(type(allocation_start) in (int,float) and math.isfinite(allocation_start) and allocation_start>0,'actual allocation epoch required')
    need(bool(re.fullmatch('[1-9][0-9]*',job_id)),'Slurm job ID required')
    row=dict(schema='z1_depth_global_clock031',manifest_sha256=manifest_sha,
        qualification_job_id=job_id,first_allocation_epoch=allocation_start,
        deadline_epoch=allocation_start+28800,clock_domain='Vega_Slurm_epoch')
    write_once(path,row)
    return row


def admit_time(clock, *, now, allocation_end, bound):
    need(clock.get('schema')=='z1_depth_global_clock031' and clock.get('clock_domain')=='Vega_Slurm_epoch', '031 global clock required')
    need(clock['deadline_epoch']==clock['first_allocation_epoch']+28800,'global deadline drift')
    need(all(type(v) in (int,float) and math.isfinite(v) for v in (now,allocation_end,bound)) and bound>0,'finite clock bounds required')
    deadline=min(clock['deadline_epoch'],allocation_end)
    need(clock['first_allocation_epoch']<=now and now+bound<=deadline,'HOLD_UNCLAIMED: insufficient allocation/global time')
    return deadline


def admit_evaluation_time(manifest,clock,allocation,grant,*,now):
    need(clock.get('schema')=='z1_depth_global_clock031' and clock.get('clock_domain')=='Vega_Slurm_epoch'
         and clock.get('manifest_sha256')==digest(manifest),'031 evaluation global clock required')
    first=clock['first_allocation_epoch']
    need(type(first) in (int,float) and math.isfinite(first) and first>0
         and clock['deadline_epoch']==first+28800,'global deadline drift')
    cutoff=first+64800-180
    need(allocation.get('deadline_epoch')==cutoff and allocation.get('time_limit_seconds')==8100,
         'CPU effective scheduler Deadline/TimeLimit differs')
    bounds=(now,allocation['end_epoch'],grant['expires_epoch'])
    need(all(type(value) in (int,float) and math.isfinite(value) for value in bounds),'finite CPU clock bounds required')
    deadline=min(cutoff,allocation['end_epoch'],grant['expires_epoch'])
    need(first<=now and now+7200+60<=deadline,'HOLD_UNCLAIMED: CPU case plus finalization cannot fit')
    return deadline


def slurm_job(job_id):
    need(bool(re.fullmatch('[1-9][0-9]*',job_id)),'exact scalar Slurm ID required')
    env=dict(os.environ,TZ='UTC')
    result=subprocess.run(['scontrol','show','job','-o',job_id],capture_output=True,text=True,timeout=10,env=env)
    if result.returncode==0:
        values=dict(re.findall(r'(\w+)=([^\s]+)',result.stdout))
    else:
        result=subprocess.run(['sacct','-X','-j',job_id,'--noheader','--parsable2',
            '--format=JobIDRaw,JobName%100,State,ExitCode,User,Comment%100,Start,End'],
            check=True,capture_output=True,text=True,timeout=10,env=env)
        rows=[line.split('|') for line in result.stdout.splitlines() if line.split('|')[0]==job_id]
        need(len(rows)==1 and len(rows[0])>=8,'exact terminal accounting record required')
        values=dict(zip(('JobId','JobName','JobState','ExitCode','UserId','Comment','StartTime','EndTime'),rows[0]))
    need(values.get('JobId')==job_id,'Slurm identity mismatch')
    return values


def allocation(job_id, expected_name):
    from datetime import datetime, timezone
    remote_host();row=slurm_job(job_id)
    need(row.get('JobName')==expected_name and row.get('UserId','').split('(')[0]=='eunikhilr','foreign allocation refused')
    need(row.get('JobState')=='RUNNING' and os.environ.get('SLURM_JOB_ID')==job_id,'running own allocation required')
    need(row.get('Requeue')=='0' and row.get('Restarts','0')=='0','requeue/restart prohibited')
    require_resource_shape(row,expected_name)
    parse=lambda key:datetime.fromisoformat(row[key]).replace(tzinfo=timezone.utc).timestamp()
    result=dict(job_id=job_id,name=expected_name,start_epoch=parse('StartTime'),end_epoch=parse('EndTime'),owner='eunikhilr')
    if expected_name.startswith('depth031-slot-'):
        need(row.get('TimeLimit')=='04:30:00' and row.get('Deadline') not in (None,'N/A','Unknown'),'fixed slot TimeLimit and Slurm Deadline required')
        result.update(deadline_epoch=parse('Deadline'),time_limit_seconds=16200)
    if expected_name.startswith('depth031-eval-'):
        need(row.get('Deadline') not in (None,'','N/A','Unknown'),'CPU Slurm Deadline required')
        result.update(deadline_epoch=parse('Deadline'),time_limit_seconds=8100)
    if expected_name=='depth031-qualification' or expected_name.startswith('depth031-slot-'):
        result.update(gpu_allocation_identity())
    return result


def qualification_terminal(row):
    return row.get('JobState')=='COMPLETED' and row.get('ExitCode')=='0:0'


def ready_slots(submitted, anchors):
    need(set(submitted)<=set(range(6)) and set(anchors)<={0,4,8},'closed finite slot/anchor scope required')
    return [s for s in range(6) if s not in submitted and (s%2==0 or 4*(s//2) in anchors)]


def validate_grant(manifest, grant, *, now, scope):
    need(type(grant) is dict and set(grant)=={'schema','manifest_sha256','issued_epoch','expires_epoch','scope','issuer','nonce','review_refs'},'closed fresh grant required')
    need(grant['schema']=='z1_depth_root_grant031' and grant['manifest_sha256']==digest(manifest)
         and grant['issuer']=='root' and grant['scope']==scope,'new031 root authority required')
    need(bool(re.fullmatch('[0-9a-f]{32}',grant['nonce'])),'fresh authority nonce required')
    need(grant['issued_epoch']<=now<grant['expires_epoch'] and grant['expires_epoch']-grant['issued_epoch']<=64800+28800,'expired/overlong root authority')
    need(type(grant['review_refs']) is list and len(grant['review_refs'])>=2,'independent source/spec and resources review refs required')
    for ref in grant['review_refs']:verify_ref(ref)
    return grant


def private_environment(private, *, cpu=False):
    private=Path(private);private.mkdir(exist_ok=False)
    env=dict(os.environ)
    paths={'HOME':'home','TMPDIR':'tmp','XDG_CACHE_HOME':'cache','TORCH_HOME':'cache/torch',
           'TRITON_CACHE_DIR':'cache/triton','CUDA_CACHE_PATH':'cache/cuda','MPLCONFIGDIR':'cache/mpl',
           'WANDB_DIR':'wandb','PYTHONPYCACHEPREFIX':'pycache'}
    for key,relative in paths.items():
        path=private/relative;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    (private/'cache/warp').mkdir()
    env.update(LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2' if cpu else '4',
               MKL_NUM_THREADS='2' if cpu else '4',OPENBLAS_NUM_THREADS='2' if cpu else '4',
               NUMEXPR_NUM_THREADS='2' if cpu else '4',VECLIB_MAXIMUM_THREADS='2' if cpu else '4',WANDB_MODE='disabled',MUJOCO_GL='egl')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    if cpu:env['CUDA_VISIBLE_DEVICES']=''
    return env


def validate_owned_job(entry, row, manifest_sha):
    need(entry['manifest_sha256']==manifest_sha and row.get('JobId')==entry['job_id']
         and row.get('JobName')==entry['name'] and row.get('UserId','').split('(')[0]=='eunikhilr'
         and row.get('Comment')=='depth031:'+manifest_sha,'job ownership uncertain; no cancellation permitted')
    return entry


def cancel_expired_gpu_jobs(entries, clock, *, now, lookup=slurm_job, run=subprocess.run):
    need(now>=clock['deadline_epoch'],'deadline not reached')
    cancelled=[]
    for entry in entries:
        if entry['kind']!='training':continue
        row=lookup(entry['job_id']);validate_owned_job(entry,row,clock['manifest_sha256'])
        if row.get('JobState') in ('RUNNING','PENDING','CONFIGURING','COMPLETING'):
            run(['scancel',entry['job_id']],check=True,timeout=10)
            cancelled.append(entry['job_id'])
    return cancelled


# Narrow audited UUID helpers; no old release/registry authority imported.
def gpu_allocation_identity():
    job, visible = os.environ.get('SLURM_JOB_ID',''), os.environ.get('CUDA_VISIBLE_DEVICES','')
    need(re.fullmatch('[1-9][0-9]*',job) and re.fullmatch('(?:[0-9]+|GPU-[0-9a-fA-F-]+)',visible), 'one exact Slurm GPU allocation required')
    r = subprocess.run(['nvidia-smi','-i',visible,'--query-gpu=uuid','--format=csv,noheader'], capture_output=True,text=True,timeout=10)
    need(r.returncode == 0 and len(r.stdout) <= 1024 and len(r.stderr) <= 1024, 'GPU identity query failed')
    uuid = r.stdout.strip()
    need(re.fullmatch('GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',uuid), 'single physical GPU UUID required')
    return dict(slurm_job_id=job, gpu_uuid=uuid, cuda_visible_devices=visible, hostname=os.uname().nodename)




def verify_cuda_device(torch, identity):
    need(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.current_device()==0,
         'one available CUDA device at logical index0 required')
    value=getattr(torch.cuda.get_device_properties(0), 'uuid', None)
    if not isinstance(value, (str,bytes)):
        # PyTorch Module.cpp binds _CUuuid.bytes as vector<uint8_t> (a Python list).
        raw=getattr(value,'bytes',None)
        if type(raw) in (list,tuple):
            need(len(raw)==16 and all(type(x) is int and 0<=x<=255 for x in raw),
                 'actual CUDA UUID vector requires exactly16 uint8 integers')
            raw=bytes(raw)
        need(isinstance(raw,(bytes,bytearray,memoryview)) and len(raw)==16,
             'actual CUDA UUID object requires exactly16 raw bytes')
        value=bytes(raw)
    if isinstance(value, bytes):
        value=str(uuid.UUID(bytes=value)) if len(value)==16 else value.decode('ascii')
    need(type(value) is str, 'actual CUDA device UUID required')
    actual=uuid.UUID(value.removeprefix('GPU-'))
    need(actual==uuid.UUID(identity['gpu_uuid'].removeprefix('GPU-')), 'actual CUDA device differs from allocation UUID')
    return dict(logical_device=0,gpu_uuid='GPU-'+str(actual))




def require_resource_shape(row,name):
    from .manifest import SLURM_ACCOUNT,SLURM_QOS
    gpu=name=='depth031-qualification' or name.startswith('depth031-slot-')
    controller=name=='depth031-controller'
    cpus,memory,limit=(4,32,'00:45:00' if name=='depth031-qualification' else '04:30:00') if gpu else ((1,2,'18:00:00') if controller else (2,8,'02:15:00'))
    need(row.get('Account')==SLURM_ACCOUNT and row.get('QOS')==SLURM_QOS and row.get('Partition')==('gpu' if gpu else 'cpu'),'Slurm account/QOS/partition differs')
    need(row.get('NumCPUs')==str(cpus) and row.get('NumNodes')=='1' and row.get('TimeLimit')==limit,'Slurm CPU/node/time allocation differs')
    tres=dict(item.split('=',1) for item in row.get('AllocTRES',row.get('TRES','')).split(',') if '=' in item)
    mem=tres.get('mem','');match=re.fullmatch(r'(\d+)([MGT])',mem)
    need(match is not None,'Slurm memory reservation required')
    actual=int(match[1])*{'M':1024**2,'G':1024**3,'T':1024**4}[match[2]]
    need(actual==memory*1024**3 and tres.get('cpu')==str(cpus),'Slurm memory/CPU TRES differs')
    gpu_values=[value for key,value in tres.items() if key=='gres/gpu']
    need(gpu_values==(['1'] if gpu else []),'exact GPU reservation differs')
