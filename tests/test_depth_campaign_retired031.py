"""The historical launcher cannot start work; frozen evidence stays readable."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('args', [['--help'], ['qualify'], ['controller'], ['evaluation']])
def test_old_launcher_is_retired(direct, args):
    entry = [str(ROOT / 'scripts/z1_depth_campaign031/__main__.py')] if direct else ['-m', 'scripts.z1_depth_campaign031']
    result = subprocess.run([sys.executable, '-B', *entry, *args], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == (0 if args == ['--help'] else 2)
    assert 'retired' in (result.stdout + result.stderr).lower()
    assert 'docs/experiments/z1-depth-rewards.md' in result.stdout + result.stderr


def test_frozen_preservation_code_still_imports_independently():
    source = ROOT / '.superpowers/sdd/2026-11-01-z1-depth-reward031/prepared031-v9/source'
    if not source.is_dir():
        pytest.skip('historical frozen source is local evidence, not a checkout dependency')
    program = ('import sys;sys.path.insert(0,' + repr(str(source)) + ');'
               'from scripts.z1_depth_campaign031 import preservation;'
               'assert preservation.__file__.startswith(' + repr(str(source)) + ');'
               'assert "torch" not in sys.modules')
    subprocess.run([sys.executable, '-B', '-I', '-S', '-c', program],
                   cwd=ROOT, check=True, capture_output=True, text=True, timeout=10)
