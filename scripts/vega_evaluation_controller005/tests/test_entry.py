import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class SourceEntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name).resolve()
        self.package=self.root/'source'; self.package.mkdir()
        for name in ('entry.py','journal.py','scheduler.py','coordinator.py','controller.sbatch'):
            if (ROOT/name).exists(): shutil.copyfile(ROOT/name,self.package/name)
        self.manifest={'schema':'z1_vega_controller_source_v1','sources':{}}
        for p in self.package.iterdir():
            self.manifest['sources'][p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        self.manifest_path=self.package/'source-manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest,sort_keys=True))
    def tearDown(self): self.tmp.cleanup()
    def invoke(self):
        return subprocess.run([sys.executable,'-B','-I','-S',str(self.package/'entry.py'),'--check-source',str(self.manifest_path),'--manifest-sha256',hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()],capture_output=True,text=True)
    def test_source_only_entry_admits_exact_manifest_without_import_fallback(self):
        result=self.invoke()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'SOURCE_VERIFIED')
    def test_rebound_extra_bytecode_and_symlink_source_fail(self):
        for mode in ('changed','extra','bytecode','symlink'):
            with self.subTest(mode=mode):
                target=self.package/'extra.py'
                if mode=='changed':
                    saved=(self.package/'journal.py').read_bytes(); (self.package/'journal.py').write_bytes(saved+b'\n')
                elif mode=='extra': target.write_text('raise RuntimeError("should not load")')
                elif mode=='bytecode': target=self.package/'extra.pyc'; target.write_bytes(b'bad')
                else: target.symlink_to('/tmp')
                result=self.invoke(); self.assertNotEqual(result.returncode,0)
                if mode=='changed': (self.package/'journal.py').write_bytes(saved)
                else: target.unlink()
    def test_activation_cannot_select_unbound_adapter_or_commands(self):
        result=subprocess.run([sys.executable,'-B','-I','-S',str(self.package/'entry.py'),'--activate','--adapter','os.system'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertFalse((self.root/'owner.lease').exists())
