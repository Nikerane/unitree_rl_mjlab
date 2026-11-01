"""031 source ownership and the two Torch virtual registration owners."""
from pathlib import Path
import re
import sys
from .files import need,regular_path,safe_path,sha_file


ROLES=('mjlab','mujoco','mujoco_warp','rsl_rl','torch')


def runtime_site(manifest):
    """Locate the install from its manifest pin, never imported metadata."""
    roots=[Path(p) for p in manifest['runtime_files'] if p.endswith('/torch/__init__.py')]
    need(len(roots)==1,'one pinned Torch root required')
    site=roots[0].parent.parent
    prefix=Path(manifest['python']['path']).parent.parent
    need(site.is_absolute() and site.name=='site-packages' and re.fullmatch(r'python3\.\d+',site.parent.name)
         and site.parent.parent==prefix/'lib','runtime install differs from pinned Python')
    safe_path(site)
    for path in map(Path,manifest['runtime_files']):
        if any(str(path).endswith('/'+role+'/__init__.py') for role in ROLES):
            need(path.parent.parent==site,'runtime package roots differ')
    return site


def _pinned_runtime(manifest,path):
    need(str(path) in manifest['runtime_files'] and sha_file(path)==manifest['runtime_files'][str(path)],
         'virtual runtime implementation bytes differ: '+str(path))


def _virtual_torch(manifest,site,name,module):
    """Inspect existing registries only; dynamic getattr would create namespaces."""
    owner='classes' if name=='torch.classes' else 'ops'
    impl_name='torch._'+owner
    parent=sys.modules.get('torch');impl=sys.modules.get(impl_name)
    need(parent is not None and impl is not None,'virtual Torch owner missing: '+name)
    impl_path=site/'torch'/('_'+owner+'.py')
    need(vars(parent).get('__file__')==str(site/'torch/__init__.py')
         and vars(impl).get('__file__')==str(impl_path),'virtual Torch owner origin differs: '+name)
    _pinned_runtime(manifest,site/'torch/__init__.py');_pinned_runtime(manifest,impl_path)
    registry=vars(impl).get(owner)
    cls=vars(impl).get('_Classes' if owner=='classes' else '_Ops')
    need(cls is not None and type(registry) is cls and vars(parent).get(owner) is registry
         and sys.modules.get('torch.'+owner) is registry,'virtual Torch registry differs: '+name)
    if name=='torch.'+owner:
        expected=cls;pseudo='_'+owner+'.py'
        need(module is registry,'virtual Torch root differs: '+name)
    else:
        suffix=name.removeprefix('torch.ops.')
        need('.' not in suffix and suffix and vars(registry).get(suffix) is module,
             'virtual Torch child registration differs: '+name)
        expected=vars(impl).get('_HigherOrderNamespace' if suffix=='higher_order' else '_OpNamespace')
        pseudo='torch.ops'
        if suffix!='higher_order':need(vars(module).get('name')==suffix,'virtual Torch namespace differs: '+name)
    need(expected is not None and type(module) is expected and vars(module).get('__name__')==name
         and getattr(module,'__file__',None)==pseudo and vars(module).get('__spec__') is None,
         'virtual Torch metadata differs: '+name)


def verify_import_origins(manifest):
    source=Path(manifest['source_root']);site=runtime_site(manifest)
    _pinned_runtime(manifest,site/'torch/__init__.py')
    for name,module in tuple(sys.modules.items()):
        origin=getattr(module,'__file__',None)
        spec_origin=getattr(getattr(module,'__spec__',None),'origin',None)
        scientific=name in ('src','scripts') or name.startswith(('src.','scripts.'))
        runtime=any(name==role or name.startswith(role+'.') for role in ROLES)
        virtual=name in ('torch.ops','torch.classes') or name.startswith('torch.ops.')
        if (scientific or runtime) and not virtual and spec_origin not in (None,'built-in','frozen'):
            need(origin is None or origin==spec_origin,'file/spec origin differs: '+name)
            origin=spec_origin
        if name=='scripts' or name=='src' or name.startswith(('src.','scripts.')):
            if origin is None:
                expected=source/Path(*name.split('.'))
                portions=getattr(module,'__path__',())
                need(list(map(str,portions))==[str(expected)] and expected.is_dir()
                     and any(p.startswith(str(expected.relative_to(source))+'/') for p in manifest['source_files']),
                     'ambient source namespace: '+name)
                locations=getattr(getattr(module,'__spec__',None),'submodule_search_locations',None)
                need(locations is None or list(map(str,locations))==[str(expected)],
                     'source namespace spec differs: '+name)
                safe_path(expected)
            else:
                path=Path(origin)
                need(path.is_absolute() and '..' not in path.parts and path.is_relative_to(source)
                     and sha_file(path)==manifest['source_files'].get(str(path.relative_to(source))),
                     'scientific origin escaped freeze: '+name)
        elif name in ('torch.ops','torch.classes') or name.startswith('torch.ops.'):
            _virtual_torch(manifest,site,name,module)
        elif any(name==role or name.startswith(role+'.') for role in ROLES) and origin:
            path=Path(origin)
            need(path.is_absolute() and '..' not in path.parts and path.is_relative_to(site),
                 'runtime origin escaped pinned install: '+name)
            need(path.is_file(),'regular runtime file required: '+name)
            regular_path(path)
            if str(path) in manifest['runtime_files']:
                need(sha_file(path)==manifest['runtime_files'][str(path)],'runtime origin bytes differ: '+name)
