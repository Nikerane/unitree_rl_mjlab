"""Hash-pinned source-only loader for the frozen002 archive validators."""
import hashlib
import os
from pathlib import Path
import sys
import stat
import threading
import types

_HASHES = {
    'archive_contract.py': '1f228baf0c004864c7974107f0792e710b70e18c6edbcd83bcd31f14c9f3ac24',
    'remote_policy_archive.py': '835af461be2ded473c7bcbe2d4bc9e9a30d26a188ab558b7442177a94c2cb9e7',
    'policy_archive.py': 'c8e093626dc07ca77aae7b326895f7f130185aed24f822fdce04a51c98ebcc25',
}
_LOCK = threading.RLock()
_MODULES = None


def load_modules():
    """Compile only bounded, pinned sources; never consume cached bytecode."""
    global _MODULES
    with _LOCK:
        sources = {}
        for name, expected in _HASHES.items():
            path = Path(__file__).parent / name
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 131072:
                os.close(fd)
                raise ValueError('bounded regular compatibility source required: ' + name)
            with os.fdopen(fd, 'rb') as stream:
                data = stream.read(131073)
            if len(data) > 131072 or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('compatibility source size/SHA differs: ' + name)
            sources[name] = data
        if _MODULES is None:
            result = {}
            sentinel = object()
            previous = sys.modules.get('archive_contract', sentinel)
            try:
                for key, filename in [('contract', 'archive_contract.py'),
                                      ('policy', 'policy_archive.py'),
                                      ('remote', 'remote_policy_archive.py')]:
                    module = types.ModuleType(__name__ + '.' + filename[:-3])
                    module.__file__ = str(Path(__file__).parent / filename)
                    exec(compile(sources[filename], module.__file__, 'exec'), module.__dict__)
                    result[key] = module
                    if key == 'contract':
                        sys.modules['archive_contract'] = module
                _MODULES = result
            finally:
                if previous is sentinel:
                    sys.modules.pop('archive_contract', None)
                else:
                    sys.modules['archive_contract'] = previous
        return dict(_MODULES)
