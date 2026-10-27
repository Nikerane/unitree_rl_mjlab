"""Nonblocking owned process pipes with one outer wall deadline.

No threads, archive buffers, shell input, local tar files or unbounded pipe IO.
The consumer receives only bounded nonseek read(size), and its result is usable
only after stdin closed, stdout/stderr EOF and a successful reaped child exit.
"""
from __future__ import annotations
import hashlib
import math
import os
import selectors
import signal
import subprocess
import time

CHUNK=1024**2


def _require(value,message):
    if not value: raise ValueError(message)


def _check(deadline):
    if time.monotonic()>=deadline: raise TimeoutError('owned transport wall deadline expired')


class _Pipes:
    def __init__(self,process,payload,deadline,stdout_cap,stderr_cap):
        self.process=process;self.payload=memoryview(payload);self.deadline=deadline
        self.caps=dict(stdout=stdout_cap,stderr=stderr_cap);self.counts=dict(stdout=0,stderr=0)
        self.sha=hashlib.sha256();self.buffer=bytearray();self.stderr=bytearray()
        self.selector=selectors.DefaultSelector();self.eof=False
        for name in ('stdin','stdout','stderr'):
            stream=getattr(process,name);os.set_blocking(stream.fileno(),False)
            if name=='stdin' and not self.payload: stream.close();continue
            self.selector.register(stream,selectors.EVENT_WRITE if name=='stdin' else selectors.EVENT_READ,name)

    def pump(self,need_output=True):
        _check(self.deadline)
        events=self.selector.select(min(.1,max(0,self.deadline-time.monotonic())))
        for key,_ in events:
            name=key.data;stream=key.fileobj
            _check(self.deadline)
            if name=='stdin':
                try: count=os.write(stream.fileno(),self.payload[:65536])
                except BrokenPipeError: raise ValueError('owned child closed request stdin early')
                except BlockingIOError: continue
                _require(count>0,'zero stdin pipe write');self.payload=self.payload[count:]
                if not self.payload: self.selector.unregister(stream);stream.close()
            else:
                # Do not accumulate unread stdout while draining stderr. A
                # consumer read takes at most CHUNK, pipe backpressure is safe.
                if name=='stdout' and len(self.buffer)>=CHUNK: continue
                maximum=min(65536,CHUNK-len(self.buffer)) if name=='stdout' else 65536
                try: data=os.read(stream.fileno(),maximum)
                except BlockingIOError: continue
                if not data:
                    self.selector.unregister(stream)
                    if name=='stdout': self.eof=True
                    continue
                self.counts[name]+=len(data)
                _require(self.counts[name]<=self.caps[name],name+' byte cap exceeded')
                if name=='stdout': self.buffer.extend(data);self.sha.update(data)
                else: self.stderr.extend(data)
        _check(self.deadline)

    def read(self,size):
        _require(type(size) is int and 0<=size<=CHUNK,'bounded positive read size required')
        if not size:return b''
        while len(self.buffer)<size and not self.eof:self.pump()
        result=bytes(self.buffer[:size]);del self.buffer[:size]
        return result

    def finish(self):
        # A consumer must exhaust binary output; no hidden trailing byte allowed.
        _require(not self.buffer,'consumer left unread stdout')
        while self.selector.get_map():
            self.pump()
            _require(not self.buffer,'consumer did not read transport EOF')
        while self.process.poll() is None:
            _check(self.deadline)
            try:self.process.wait(timeout=min(.05,max(.001,self.deadline-time.monotonic())))
            except subprocess.TimeoutExpired:pass
        _check(self.deadline)
        _require(self.process.returncode==0,'owned child exit '+str(self.process.returncode))


def _reap(process):
    # Kill the whole group, including descendants retaining pipe descriptors,
    # even when the leader has already exited. Reap the owned leader always.
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    try:process.wait(timeout=1)
    except subprocess.TimeoutExpired:raise RuntimeError('owned process could not be reaped')


def run_process(argv,payload,*,deadline,stdout_cap,stderr_cap=CHUNK,stdin_cap=8*CHUNK,consumer=None):
    _require(type(deadline) in (int,float) and math.isfinite(deadline),'finite monotonic deadline required')
    _check(deadline);deadline=min(deadline,time.monotonic()+600)
    for value,hard in ((stdout_cap,201326592),(stderr_cap,CHUNK),(stdin_cap,8*CHUNK)):
        _require(type(value) is int and 0<value<=hard,'finite pipe byte cap required')
    _require(isinstance(payload,bytes) and len(payload)<=stdin_cap,'request stdin cap exceeded')
    _require(isinstance(argv,list) and all(isinstance(x,str) for x in argv),'explicit argv required')
    process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                             start_new_session=True,bufsize=0)
    pipes=None
    try:
        pipes=_Pipes(process,payload,deadline,stdout_cap,stderr_cap)
        if consumer is None:
            chunks=[]
            while True:
                block=pipes.read(CHUNK)
                if not block:break
                chunks.append(block)
            value=None;stdout=b''.join(chunks)
        else:value=consumer(pipes);stdout=None
        pipes.finish()
        return dict(returncode=process.returncode,stdout=stdout,value=value,stdout_bytes=pipes.counts['stdout'],
                    stderr_bytes=pipes.counts['stderr'],stderr=bytes(pipes.stderr),stdout_sha256=pipes.sha.hexdigest())
    except BaseException:
        _reap(process);raise
    finally:
        if pipes is not None:pipes.selector.close()
        for stream in (process.stdin,process.stdout,process.stderr):
            if not stream.closed:stream.close()
