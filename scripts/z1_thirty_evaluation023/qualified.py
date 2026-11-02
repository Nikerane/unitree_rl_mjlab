"""Load exactly preserved sources; qualified physics and measurement math stay unchanged."""
import hashlib
import itertools
from pathlib import Path
import sys
import types
from .files import need, read_json, regular_path, sha_file

HERE = Path(__file__).resolve().parent
VENDOR = HERE/'vendor'
_READER_IDS = itertools.count()


def verify_copies():
    manifest = read_json(HERE/'COPIED-SOURCES023.json')
    actual = {str(p.relative_to(VENDOR)) for p in VENDOR.rglob('*') if p.is_file()}
    need(actual == set(manifest), 'copied source closure differs')
    for name, record in manifest.items():
        need(sha_file(VENDOR/name) == record['sha256'], 'copied source SHA differs: '+name)
    return {name:record['sha256'] for name,record in manifest.items()}


def load_pinned(path, expected, name):
    body = regular_path(path).read_bytes()
    need(hashlib.sha256(body).hexdigest() == expected, 'module SHA differs: '+str(path))
    need(name not in sys.modules, 'ambient module name refused: '+name)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(body,str(path),'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(name,None)
        raise
    return module


def pilot_utilities():
    pins = verify_copies()
    for name in ('pilot_contract','pilot_integration'):
        relative = 'pilot-compat001/'+name+'.py'
        module = load_pinned(VENDOR/relative,pins[relative],name)
    return module


def measurement_reader():
    pins = verify_copies()
    relative = 'sealer001/measurement_reader.py'
    return load_pinned(VENDOR/relative,pins[relative],'thirty_measurement_reader023_'+str(next(_READER_IDS)))
