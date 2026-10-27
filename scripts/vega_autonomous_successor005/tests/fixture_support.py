"""Test-only disposable root/platform translations; never used by production.

The tested children report their actual platform/PID/host. No platform spoofing
or sys.platform monkeypatch occurs. Only copied validation constants translate
Linux to the actual local host for pure macOS IO execution.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
OLD=REPO/'scripts/vega_policy_archive/tests'
FROZEN=REPO/'.superpowers/sdd/2026-10-23-z1-overnight-execution/evaluation-staged-integration-repair-proposal002'
CPU='/cephhome/eunikhilr/campaigns/z1-horizon-evaluation-20261027-staged-integration002'
BACKUP='/cephhome/eunikhilr/campaign-backups/z1-horizon-evaluation-20261027-autonomous005'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def body(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body(value));return dict(path=str(path),sha256=sha(path))
def load(path,name,location=None):
    import types
    value=types.ModuleType(name);value.__file__=str(location or path);exec(compile(Path(path).read_bytes(),str(path),'exec'),value.__dict__);return value

def manifest(root):
    return write(root/'source-manifest.json',dict(schema='z1_vega_policy_archive_package_v1',entrypoint='independent_verify.py',files={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*.py')) if 'tests' not in p.relative_to(root).parts}))

def translated_package(base):
    root=base/'helper';shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('tests','SDD','__pycache__'))
    cpu=str(base/'cpu');backup=str(base/'backup')
    for path in root.rglob('*.py'):
        source=path.read_text().replace(CPU,cpu).replace(BACKUP,backup)
        source=source.replace("'/cephhome/eunikhilr'",repr(str(base)))
        source=source.replace("'linux'",repr(sys.platform))
        source=source.replace("'/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/bin/python'",repr(sys.executable))
        # Preserve real current quota guard logic but translate its actual quota
        # source in this source-only fixture, as no Ceph exists on the Mac.
        source=source.replace("maximum=int(os.getxattr(root,'ceph.quota.max_bytes'))",'maximum=10**12').replace("used=int(os.getxattr(root,'ceph.dir.rbytes'))",'used=0')
        path.write_text(source)
    compat=root/'archive/compat/__init__.py';s=compat.read_text()
    for name in ('archive_contract.py','policy_archive.py','remote_policy_archive.py'):
        s=s.replace(sha(ROOT/'archive/compat'/name),sha(root/'archive/compat'/name))
    compat.write_text(s)
    reference=manifest(root)
    return root,cpu,backup,reference

def generic(base):
    root,cpu,backup,reference=translated_package(base)
    r=load(root/'reservation.py','fixture_reservation');control=r.CONTROL_ROOT
    ledger=r.zero_ledger(preissue=dict(schema='z1_vega_preissue_descriptor_v1',kind='CONTROL_ONLY'),layout_sha256='c'*64)
    ledger_ref=write(Path(control)/'ledger.json',ledger)
    authority=write(Path(control)/'preservation-authority.json',dict(schema='z1_vega_control_preservation_authority_v1',backend_mode='VEGA_RESIDENT_FULL_ARCHIVE_VEGA_PROCESS_V1',helper_package=reference,backup_root=backup,control_root=control))
    bundle=Path(control)/'input-bundles/source';(bundle/'controls/empty').mkdir(parents=True);(bundle/'private').mkdir()
    (bundle/'controls/control.json').write_bytes(body(dict(important='original bytes',historical='/Users/historical-only')))
    (bundle/'private/metadata').write_bytes(b'private original\x00')
    request=dict(schema='z1_horizon_vega_resident_archive_request_v1',kind='CONTROL_ONLY',backend_mode='VEGA_RESIDENT_FULL_ARCHIVE_VEGA_PROCESS_V1',helper_package_reference=reference,remote_helper_package_reference=reference,preservation_authority=authority,control_root=control,control_id='source',source_root=str(bundle/'controls'),private_root=str(bundle/'private'),remote_archive_destination=backup+'/control-only/source',capacity_ledger_reference=ledger_ref,max_source_bytes=64*1024**2,max_archive_bytes=64*1024**2,max_metadata_bytes=4194304,max_evidence_bytes=8388608,max_members=8192,wall_seconds=120,future_reserve_bytes=11274289152,remote_future_reserve_bytes=11274289152)
    request_ref=write(Path(control)/'request.json',request)
    return root,request,request_ref
