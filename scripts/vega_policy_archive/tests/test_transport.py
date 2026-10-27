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
        with mock.patch.object(resident_archive,'BOOTSTRAP',"import time\ndef load_verified_package(reference): time.sleep(20)\n"):
            program=resident_archive.remote_program(dict(path='/unused',sha256='a'*64),1)
        start=time.monotonic()
        result=__import__('subprocess').run([sys.executable,'-B','-I','-S','-c',program],input=b'',capture_output=True,timeout=3)
        self.assertNotEqual(result.returncode,0);self.assertIn(b'outer remote bootstrap deadline',result.stderr)
        self.assertLess(time.monotonic()-start,2)
