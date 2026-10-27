"""Actual owned child pipes; never contacts SSH or another host."""
import importlib
import os
from pathlib import Path
import sys
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
try: transport=importlib.import_module('vega_policy_archive.transport')
except ModuleNotFoundError: transport=None

class TransportTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(transport,'bounded transport is not implemented')
    def run_child(self,source,payload=b'',**kw):
        return transport.run_process([sys.executable,'-B','-I','-S','-c',source],payload,
            deadline=time.monotonic()+kw.pop('seconds',5),stdout_cap=kw.pop('stdout_cap',1024**2),
            stderr_cap=kw.pop('stderr_cap',1024**2),**kw)
    def test_concurrently_drains_stderr_and_request_backpressure(self):
        body=b'x'*400000
        result=self.run_child("import sys;sys.stderr.buffer.write(b'e'*400000);sys.stderr.flush();data=sys.stdin.buffer.read();sys.stdout.buffer.write(data)",body)
        self.assertEqual(result['stdout'],body);self.assertEqual(result['stderr_bytes'],400000);self.assertEqual(result['returncode'],0)
    def test_nonseek_stream_and_exit_zero_required_after_eof(self):
        result=self.run_child("import sys;sys.stdout.buffer.write(b'abc')",consumer=lambda stream: (stream.read(2),stream.read(2),stream.read(1)))
        self.assertEqual(result['value'],(b'ab',b'c',b''))
        with self.assertRaisesRegex(ValueError,'exit'):
            self.run_child("import sys;sys.stdout.buffer.write(b'valid');sys.stdout.flush();sys.exit(1)",consumer=lambda stream: stream.read(8))
    def test_stdout_stderr_and_stdin_caps(self):
        for channel in ('stdout','stderr'):
            with self.subTest(channel=channel),self.assertRaisesRegex(ValueError,'cap'):
                self.run_child("import sys;sys.%s.buffer.write(b'x'*99999)"%channel,stdout_cap=100,stderr_cap=100)
        with self.assertRaises(ValueError): self.run_child('pass',b'123',stdin_cap=2)
    def test_timeout_kills_and_reaps_owned_process_group(self):
        start=time.monotonic()
        with self.assertRaises(TimeoutError): self.run_child('import time;time.sleep(20)',seconds=.1)
        self.assertLess(time.monotonic()-start,3)
        with self.assertRaises(TimeoutError): self.run_child('import time;time.sleep(20)',b'x'*400000,seconds=.1)
    def test_consumer_cancellation_reaps_and_bounded_read(self):
        def cancel(stream):
            with self.assertRaises(ValueError): stream.read(-1)
            raise RuntimeError('consumer cancellation')
        with self.assertRaisesRegex(RuntimeError,'cancellation'): self.run_child('import time;time.sleep(20)',consumer=cancel)
    def test_expired_deadline_and_invalid_caps(self):
        with self.assertRaises(TimeoutError): transport.run_process(['unused'],b'',deadline=time.monotonic()-1,stdout_cap=1)
        for cap in (True,-1,0):
            with self.assertRaises(ValueError): self.run_child('pass',stdout_cap=cap)

if __name__=='__main__': unittest.main()

class OuterGuardTests(unittest.TestCase):
    def test_real_blocked_kernel_read_is_interrupted_by_outer_alarm(self):
        from vega_policy_archive import resident_archive
        read_fd,write_fd=os.pipe();previous=resident_archive._watchdog(time.monotonic()+.05)
        try:
            with self.assertRaises(TimeoutError):os.read(read_fd,1)
        finally:
            resident_archive._unwatch(previous);os.close(read_fd);os.close(write_fd)
    def test_late_nonzero_after_valid_gzip_cannot_return_verification(self):
        import gzip
        data=gzip.compress(b'complete verified gzip')
        with self.assertRaisesRegex(ValueError,'exit'):
            transport.run_process([sys.executable,'-B','-I','-S','-c',"import sys;sys.stdout.buffer.write(sys.stdin.buffer.read());sys.stdout.flush();sys.exit(1)"],data,deadline=time.monotonic()+5,stdout_cap=1024,
                consumer=lambda stream:gzip.GzipFile(fileobj=stream).read())

    def test_remote_bootstrap_loading_has_an_outer_deadline(self):
        from unittest import mock
        from vega_policy_archive import resident_archive
        self.assertTrue(callable(getattr(resident_archive,'remote_program',None)),'remote bootstrap must be guarded before helper reads')
        with mock.patch.object(resident_archive,'BOOTSTRAP',resident_archive.BOOTSTRAP+"\nimport time\ndef load_verified_package(*args): time.sleep(20)\n"):
            program=resident_archive.remote_program(1)
        start=time.monotonic()
        result=__import__('subprocess').run([sys.executable,'-B','-I','-S','-c',program],input=b'{"request":{"remote_helper_package_reference":{}},"remote_wall_seconds":1}',capture_output=True,timeout=3)
        self.assertNotEqual(result.returncode,0);self.assertIn(b'outer remote bootstrap deadline',result.stderr)
        self.assertLess(time.monotonic()-start,2)

class LargeRemotePacketTests(unittest.TestCase):
    """Exercise the SSH command construction using only actual owned children."""
    def command(self,request,bootstrap=None):
        from unittest import mock
        from vega_policy_archive import resident_archive as archive
        captured={}
        def capture(argv,payload,**kwargs):
            captured.update(argv=argv,payload=payload)
            return dict(stdout=b'{}')
        with mock.patch.object(transport,'run_process',side_effect=capture):
            if bootstrap is None:archive.SSHTransport().call('inspect',request,deadline=time.monotonic()+5)
            else:
                with mock.patch.object(archive,'BOOTSTRAP',bootstrap):archive.SSHTransport().call('inspect',request,deadline=time.monotonic()+5)
        program=__import__('shlex').split(captured['argv'][-1])[-1]
        return program,captured
    def request(self):
        from vega_policy_archive import resident_archive as archive
        return dict(ssh_host='vega',remote_python=archive.REMOTE_PYTHON,max_archive_bytes=201326592,
            remote_helper_package_reference=dict(path='/unused/package-manifest.json',sha256='a'*64),
            padding='accepted-public-data:'+('x'*(512*1024)))
    def observed_bootstrap(self,marker):
        from vega_policy_archive import resident_archive as archive
        # Preserve the real trusted stdin parser; replace only the already pinned
        # package loader boundary with a harmless test observer.
        return archive.BOOTSTRAP+'''
class OnceInput:
    def __init__(self,stream):self.stream=stream;self.calls=0
    def read(self,maximum):
        self.calls+=1
        if self.calls>1:raise AssertionError('stdin read more than once')
        return self.stream.read(maximum)
observed_input=OnceInput(sys.stdin.buffer)
sys.stdin=types.SimpleNamespace(buffer=observed_input)
def load_verified_package(reference,request,base,remote=False):
    Path('''+repr(str(marker))+''').write_text('verified-loader-boundary-reached')
    request['seen_by_loader']=True
    def run(packet=None,*,outer_deadline=None):
        if packet is None or packet['request'] is not request:raise AssertionError('same parsed packet was not passed onward')
        print(json.dumps(dict(padding_bytes=len(packet['request']['padding']),reads=observed_input.calls,same_object=packet['request']['seen_by_loader'])))
    return {'vega_policy_archive.resident_archive':types.SimpleNamespace(remote_main=run)}
'''
    def test_large_admitted_public_request_does_not_expand_python_or_ssh_argv(self):
        request=self.request();program,captured=self.command(request)
        self.assertLess(len(captured['argv'][-1].encode()),64*1024,'stdin-admitted public data leaked into SSH/Python argv')
        self.assertNotIn(request['padding'],program)
        packet=__import__('json').loads(captured['payload'])
        self.assertEqual(packet['request'],request)
        self.assertGreater(len(captured['payload']),512*1024)
        self.assertLess(len(captured['payload']),8*1024**2)
    def test_owned_child_accepts_large_packet_once_and_passes_same_parsed_object(self):
        import tempfile
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'loader-reached';request=self.request()
            program,captured=self.command(request,self.observed_bootstrap(marker))
            try:result=subprocess.run([sys.executable,'-B','-I','-S','-c',program],input=captured['payload'],capture_output=True,timeout=10)
            except OSError as exc:self.fail('accepted stdin packet overflowed argv before owned child start: '+str(exc))
            self.assertEqual(result.returncode,0,result.stderr.decode())
            value=__import__('json').loads(result.stdout)
            self.assertEqual(value,dict(padding_bytes=len(request['padding']),reads=1,same_object=True))
            self.assertTrue(marker.is_file())
    def test_duplicate_nonfinite_and_oversize_packets_reject_before_loader(self):
        import tempfile
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'loader-reached';request=self.request();request['padding']='small'
            program,_=self.command(request,self.observed_bootstrap(marker))
            cases=[(b'{"request":{},"request":{},"remote_wall_seconds":5}',b'duplicate JSON key'),
                   (b'{"request":{"x":1,"x":2},"remote_wall_seconds":5}',b'duplicate JSON key'),
                   (b'{"request":{"x":NaN},"remote_wall_seconds":5}',b'nonfinite JSON'),
                   (b'x'*(8*1024**2+1),b'stdin byte cap')]
            for body,error in cases:
                if marker.exists():marker.unlink()
                result=subprocess.run([sys.executable,'-B','-I','-S','-c',program],input=body,capture_output=True,timeout=10)
                with self.subTest(error=error):
                    self.assertNotEqual(result.returncode,0)
                    self.assertFalse(marker.exists(),'candidate loader ran before trusted packet rejection')
                    self.assertIn(error,result.stderr)
