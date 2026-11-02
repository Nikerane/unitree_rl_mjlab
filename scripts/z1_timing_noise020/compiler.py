"""Closed, pinned Warp compiler configuration and file-only runtime receipts."""
import importlib
from pathlib import Path
import sys
from . import admission as a

KERNEL_CACHE_VERSION='1.14.0'
_SITE='/cephhome/eunikhilr/repos/unitree_rl_mjlab/.venv/lib/python3.12/site-packages/warp/'
RUNTIME_FILES={
    _SITE+'config.py':'ab11c753eec21886ef3979f2ea0fba12b800dbb04361c460a829167c0eabae38',
    _SITE+'_src/context.py':'84b81f6fdf4c56bd2884d9b1a13a05de4bbf9394c309aac3ed87e41e84d9b4c7',
    _SITE+'_src/build.py':'c359106daff6771cf0ca7dbe690122c846fc4daddaaf1659686fe5157bb1a5d5',
}
_ROLES={'warp.config':_SITE+'config.py','warp._src.context':_SITE+'_src/context.py',
        'warp._src.build':_SITE+'_src/build.py'}

def policy():
    return dict(schema='z1_warp_compiler_policy022',use_precompiled_headers=False,kernel_cache_version=KERNEL_CACHE_VERSION,runtime_files=dict(RUNTIME_FILES))

def _origins(warp,modules):
    actual={}
    for role,path in _ROLES.items():
        module=warp.config if role=='warp.config' else modules.get(role)
        origin=getattr(module,'__file__',None)
        a.need(type(origin) is str and origin==path,'compiler runtime origin differs: '+role)
        actual[path]=origin
    return actual

def verify_configured(release,private,*,modules=None):
    modules=sys.modules if modules is None else modules
    a.need(a.canonical(release['compiler_policy'])==a.canonical(policy()),'closed compiler policy differs')
    warp=modules.get('warp');a.need(warp is not None,'Warp configuration missing')
    origins=_origins(warp,modules)
    a.need(type(getattr(warp.config,'use_precompiled_headers',None)) is bool
           and warp.config.use_precompiled_headers is False,'Warp PCH setting drifted')
    a.need(type(getattr(warp.config,'version',None)) is str
           and warp.config.version==KERNEL_CACHE_VERSION,'Warp compiler version differs')
    cache=Path(private)/'cache/warp'
    # Pinned build.init_kernel_cache appends config.version during Runtime init.
    if getattr(modules['warp._src.context'],'runtime',None) is not None:cache=cache/KERNEL_CACHE_VERSION
    a.need(warp.config.kernel_cache_dir==str(cache),'Warp private cache/initialization phase differs')
    return origins

def configure(release,private,output,case,*,modules=None,importer=None):
    """Apply once, before scientific imports or Warp runtime initialization."""
    modules=sys.modules if modules is None else modules;importer=importlib.import_module if importer is None else importer
    a.need(a.canonical(release['compiler_policy'])==a.canonical(policy()),'closed compiler policy differs')
    a.need(not any(name=='src' or name.startswith(('src.','mjlab','mujoco_warp'))
                   or name in ('scripts.train',*a.FAMILIES.values()) for name in modules),
           'compiler configuration applied after scientific imports')
    warp=importer('warp')
    for role in ('warp._src.context','warp._src.build'):importer(role)
    _origins(warp,modules)
    a.need(getattr(modules['warp._src.context'],'runtime',None) is None,'compiler configuration applied after Warp initialization')
    a.need(type(getattr(warp.config,'version',None)) is str
           and warp.config.version==KERNEL_CACHE_VERSION,'Warp compiler version differs')
    initial=getattr(warp.config,'use_precompiled_headers',None)
    a.need(type(initial) is bool,'supported bool Warp use_precompiled_headers API required')
    warp.config.use_precompiled_headers=False
    warp.config.kernel_cache_dir=str(Path(private)/'cache/warp')
    origins=verify_configured(release,private,modules=modules)
    receipt=dict(schema='z1_warp_compiler_options022',manifest_sha256=a.digest(release),
                 compiler_policy_sha256=a.digest(release['compiler_policy']),case=case,
                 private=str(private),kernel_cache_dir=warp.config.kernel_cache_dir,kernel_cache_version=warp.config.version,
                 initial_use_precompiled_headers=initial,use_precompiled_headers=warp.config.use_precompiled_headers,
                 runtime_origins=origins,phase='BEFORE_SCIENTIFIC_IMPORTS_AND_WARP_INITIALIZATION')
    a.claim_once(Path(output)/'compiler-options.json',receipt)
    return receipt

def verify_receipt(release,case,output,private):
    """Require the actual child receipt, bound to this case and exact manifest."""
    receipt=a.read_json(Path(output)/'compiler-options.json',cap=16384)
    expected=dict(schema='z1_warp_compiler_options022',manifest_sha256=a.digest(release),
                  compiler_policy_sha256=a.digest(release['compiler_policy']),case=case,
                  private=str(private),kernel_cache_dir=str(Path(private)/'cache/warp'),kernel_cache_version=KERNEL_CACHE_VERSION,
                  initial_use_precompiled_headers=receipt.get('initial_use_precompiled_headers'),
                  use_precompiled_headers=False,runtime_origins={p:p for p in RUNTIME_FILES},
                  phase='BEFORE_SCIENTIFIC_IMPORTS_AND_WARP_INITIALIZATION')
    a.need(a.canonical(release['compiler_policy'])==a.canonical(policy()),'closed compiler policy differs')
    a.need(type(receipt.get('initial_use_precompiled_headers')) is bool
           and a.canonical(receipt)==a.canonical(expected),'bound compiler receipt differs')
    return receipt
