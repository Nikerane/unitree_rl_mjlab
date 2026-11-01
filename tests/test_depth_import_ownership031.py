"""Import-only031 boundary regressions; never construct native scientific state."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from scripts.z1_depth_campaign031 import files, worker


def isolated_bootstrap(tmp_path, *, preexisting=False, injected=False):
    source=tmp_path/'source';package=source/'scripts/z1_depth_campaign031';package.mkdir(parents=True)
    entry=package/'__main__.py'
    original=Path(worker.__file__).with_name('__main__.py')
    entry.write_bytes(original.read_bytes())
    trusted=source/'scripts/trusted031.py';trusted.write_text('VALUE=31\n')
    site=tmp_path/'site-packages';(site/'scripts').mkdir(parents=True)
    (site/'scripts/competing031.py').write_text('raise AssertionError("ambient module executed")\n')
    manifest=dict(source_root=str(source),source_files={str(p.relative_to(source)):files.sha_file(p) for p in (entry,trusted)})
    path=tmp_path/'manifest.json';files.write_once(path,manifest)
    if injected:(source/'scripts/injected031.py').write_text('raise AssertionError("unlisted code")\n')
    code=f'''
import importlib.machinery, runpy, sys, types
sys.argv=[{str(entry)!r},'--manifest',{str(path)!r},'--manifest-sha',{files.digest(manifest)!r}]
'''
    if preexisting:
        code+="sys.modules['scripts']=types.ModuleType('scripts')\n"
    code+=f'''
runpy.run_path({str(entry)!r},run_name='import_fixture031')
sys.path.append({str(site)!r})
import scripts.trusted031
assert scripts.trusted031.VALUE==31
assert list(scripts.__path__)==[{str(source/'scripts')!r}]
assert scripts.__spec__.submodule_search_locations is scripts.__path__
assert scripts.__package__=='scripts'
try:
    import scripts.competing031
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('competing module resolved')
print('TRUSTED_SCRIPTS_ONLY031')
'''
    return subprocess.run([sys.executable,'-B','-I','-S','-c',code],capture_output=True,text=True)


def test_isolated_bootstrap_owns_scripts_before_site_namespace_expands(tmp_path):
    result=isolated_bootstrap(tmp_path)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='TRUSTED_SCRIPTS_ONLY031'


def test_isolated_bootstrap_refuses_preexisting_scripts(tmp_path):
    result=isolated_bootstrap(tmp_path,preexisting=True)
    assert result.returncode!=0
    assert 'preexisting scripts package' in result.stderr


def checked(manifest, modules, monkeypatch):
    # The real checker sees only this controlled import snapshot, without imports
    # of scientific packages or mutations to the actual imported module objects.
    with monkeypatch.context() as patch:
        patch.setattr(sys,'modules',modules)
        worker.verify_import_origins(manifest)


@pytest.fixture
def import_tree(tmp_path):
    source=tmp_path/'source';source.mkdir()
    site=tmp_path/'install/lib/python3.12/site-packages';(site/'torch').mkdir(parents=True)
    manifest=dict(source_root=str(source),source_files={},runtime_files={},python={'path':str(tmp_path/'install/bin/python')})
    modules={}
    for name,filename,body in (
        ('torch','__init__.py',''),
        ('torch._ops','_ops.py',"import types\nclass _Ops(types.ModuleType):\n    __file__='_ops.py'\n    def __init__(self): super().__init__('torch.ops')\nops=_Ops()\nclass _OpNamespace(types.ModuleType):\n    __file__='torch.ops'\n    def __init__(self):\n        super().__init__('torch.ops.aten');self.name='aten'\nclass _HigherOrderNamespace(types.ModuleType):\n    __file__='torch.ops'\n    def __init__(self): super().__init__('torch.ops.higher_order')\n"),
        ('torch._classes','_classes.py',"import types\nclass _Classes(types.ModuleType):\n    __file__='_classes.py'\n    def __init__(self): super().__init__('torch.classes')\nclasses=_Classes()\n"),
    ):
        path=site/'torch'/filename;path.write_text(body)
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        modules[name]=module;manifest['runtime_files'][str(path)]=files.sha_file(path)
    for name,impl in (('ops','_ops'),('classes','_classes')):
        obj=vars(modules['torch.'+impl])[name]
        modules['torch.'+name]=obj;setattr(modules['torch'],name,obj)
    return manifest,modules,source,site


def test_verified_virtual_torch_registrations_are_not_disk_origins(import_tree,monkeypatch):
    manifest,modules,_,_=import_tree
    checked(manifest,modules,monkeypatch)


@pytest.mark.parametrize('owner',['ops','classes'])
@pytest.mark.parametrize('fault',['replacement','wrong-parent','wrong-type','file-backed','unknown'])
def test_virtual_torch_spoofs_remain_rejected(import_tree,monkeypatch,fault,owner):
    manifest,modules,_,site=import_tree
    name='torch.'+owner;impl='torch._'+owner;pseudo='_'+owner+'.py'
    if fault=='replacement':modules[name]=types.ModuleType(name)
    elif fault=='wrong-parent':setattr(modules['torch'],owner,types.ModuleType(name))
    elif fault=='wrong-type':
        obj=types.ModuleType(name);obj.__file__=pseudo
        modules[name]=obj;setattr(modules['torch'],owner,obj);setattr(modules[impl],owner,obj)
    elif fault=='file-backed':modules[name].__file__=str(site/'torch'/pseudo)
    else:
        obj=types.ModuleType('torch.unknown031');obj.__file__=pseudo;modules[obj.__name__]=obj
    with pytest.raises(ValueError):checked(manifest,modules,monkeypatch)



@pytest.mark.parametrize('fault',['source-corrupt','source-outside','source-unlisted','source-namespace',
    'scripts-namespace','runtime-corrupt','runtime-outside','runtime-traversal','runtime-relative','implementation-corrupt'])
def test_import_policy_rejects_untrusted_source_and_runtime(import_tree,monkeypatch,fault,tmp_path):
    manifest,modules,source,site=import_tree
    # Ordinary files remain checked even when there are no Torch virtual objects.
    modules={name:obj for name,obj in modules.items() if name not in ('torch.ops','torch.classes')}
    if fault.startswith('source-'):
        if fault=='source-namespace':
            obj=types.ModuleType('src.foreign');obj.__path__=[str(tmp_path/'ambient/src/foreign')]
        else:
            path=(tmp_path if fault=='source-outside' else source)/'fixture031.py';path.write_text('VALUE=1\n')
            if fault!='source-unlisted':manifest['source_files'][str(path.relative_to(source)) if path.is_relative_to(source) else 'fixture031.py']=files.sha_file(path)
            if fault=='source-corrupt':path.write_text('VALUE=2\n')
            obj=types.ModuleType('src.fixture031');obj.__file__=str(path)
        modules[obj.__name__]=obj
    elif fault=='scripts-namespace':
        obj=types.ModuleType('scripts');obj.__path__=[str(source/'scripts'),str(site/'scripts')];modules['scripts']=obj
    elif fault in ('runtime-corrupt','implementation-corrupt'):
        (site/'torch'/('_ops.py' if fault=='implementation-corrupt' else '__init__.py')).write_text('# corrupt\n')
    else:
        obj=types.ModuleType('torch.foreign031')
        obj.__file__={'runtime-outside':str(tmp_path/'foreign.py'),'runtime-traversal':str(site/'../../foreign.py'),'runtime-relative':'relative.py'}[fault]
        modules[obj.__name__]=obj
    with pytest.raises(ValueError):checked(manifest,modules,monkeypatch)


def test_verified_source_and_ordinary_runtime_files_remain_allowed(import_tree,monkeypatch):
    manifest,modules,source,site=import_tree
    modules={name:obj for name,obj in modules.items() if name not in ('torch.ops','torch.classes')}
    path=source/'fixture031.py';path.write_text('VALUE=31\n')
    obj=types.ModuleType('src.fixture031');obj.__file__=str(path);modules[obj.__name__]=obj
    manifest['source_files']['fixture031.py']=files.sha_file(path)
    extension=site/'torch/native031.so';extension.write_bytes(b'not loaded; origin-only fixture')
    obj=types.ModuleType('torch.native031');obj.__file__=str(extension);modules[obj.__name__]=obj
    checked(manifest,modules,monkeypatch)



def test_isolated_bootstrap_rejects_injected_source_before_binding_scripts(tmp_path):
    result=isolated_bootstrap(tmp_path,injected=True)
    assert result.returncode!=0
    assert 'bootstrap source closure differs' in result.stderr


@pytest.mark.parametrize('kind',['normal','higher_order'])
def test_registered_virtual_ops_children_are_verified(import_tree,monkeypatch,kind):
    manifest,modules,_,_=import_tree
    impl=modules['torch._ops']
    # Representative implementation classes remain owned by the hashed module.
    # The fixture has no tensor, native or operator access methods.
    child=vars(impl)['_HigherOrderNamespace' if kind=='higher_order' else '_OpNamespace']()
    suffix='higher_order' if kind=='higher_order' else 'aten'
    vars(modules['torch.ops'])[suffix]=child
    modules['torch.ops.'+suffix]=child
    checked(manifest,modules,monkeypatch)


def test_prepared_runtime_mapping_retains_375_and_adds_two_exact_virtual_owners(tmp_path):
    from scripts.z1_depth_campaign031 import manifest as campaign
    site=tmp_path/'install/lib/python3.12/site-packages'
    before={str(site/'torch/__init__.py'):'a'*64}
    before.update({str(site/f'torch/native{i}.so'):'b'*64 for i in range(374)})
    result=campaign.runtime_files_with_virtual_owners(before,{'path':str(tmp_path/'install/bin/python')})
    assert len(result)==377 and {p:result[p] for p in before}==before
    assert result[str(site/'torch/_ops.py')]=='c3d561d13aa4dede4525a5df8b47e38a54135653d1907101211c843df467a519'
    assert result[str(site/'torch/_classes.py')]=='0db3dc302e28fd085065ea2883ce33fa4af406b9fd1439f2a431f053e9a764b9'
    assert len(before)==375


@pytest.mark.parametrize('fault',['symlink','missing','directory','spec-outside','spec-conflict','scripts-file'])
def test_physical_import_provenance_rejects_confirmed_boundary_gaps(import_tree,monkeypatch,tmp_path,fault):
    from importlib.machinery import ModuleSpec
    manifest,modules,source,site=import_tree
    modules={name:obj for name,obj in modules.items() if name not in ('torch.ops','torch.classes')}
    obj=types.ModuleType('torch.foreign031')
    if fault=='scripts-file':
        path=tmp_path/'scripts.py';path.write_text('VALUE=0\n')
        obj=types.ModuleType('scripts');obj.__file__=str(path)
    elif fault=='spec-outside':obj.__spec__=ModuleSpec(obj.__name__,loader=None,origin=str(tmp_path/'outside.py'))
    else:
        path=site/'torch/unpinned031.py';obj.__file__=str(path)
        if fault=='symlink':
            target=tmp_path/'outside.py';target.write_text('VALUE=0\n');path.symlink_to(target)
        elif fault=='directory':path.mkdir()
        elif fault=='spec-conflict':
            path.write_text('VALUE=0\n');obj.__spec__=ModuleSpec(obj.__name__,loader=None,origin=str(tmp_path/'outside.py'))
    modules[obj.__name__]=obj
    with pytest.raises(ValueError):checked(manifest,modules,monkeypatch)


@pytest.mark.parametrize('fault',['unregistered','wrong-name','descendant'])
def test_virtual_ops_child_requires_exact_existing_registration(import_tree,monkeypatch,fault):
    manifest,modules,_,_=import_tree
    child=vars(modules['torch._ops'])['_OpNamespace']()
    if fault!='unregistered':vars(modules['torch.ops'])['aten']=child
    name='torch.ops.aten'
    if fault=='wrong-name':child.name='foreign'
    if fault=='descendant':name='torch.ops.aten.foreign'
    modules[name]=child
    with pytest.raises(ValueError):checked(manifest,modules,monkeypatch)


def test_real_installed_torch_virtual_objects_pass_isolated_policy_without_operations(tmp_path):
    import sysconfig
    from scripts.z1_depth_campaign031 import manifest as campaign
    source=tmp_path/'source';package=source/'scripts/z1_depth_campaign031';package.mkdir(parents=True)
    for name in ('__main__.py','__init__.py','import_policy.py','files.py'):
        (package/name).write_bytes(Path(worker.__file__).with_name(name).read_bytes())
    site=Path(sysconfig.get_paths()['purelib'])
    runtime=campaign.runtime_files_with_virtual_owners({str(site/'torch/__init__.py'):files.sha_file(site/'torch/__init__.py')},{'path':sys.executable})
    manifest=dict(source_root=str(source),source_files={str(p.relative_to(source)):files.sha_file(p) for p in package.iterdir()},
        runtime_files=runtime,python={'path':sys.executable})
    path=tmp_path/'manifest.json';files.write_once(path,manifest);entry=package/'__main__.py'
    code=f'''
import json,runpy,sys
sys.argv=[{str(entry)!r},'--manifest',{str(path)!r},'--manifest-sha',{files.digest(manifest)!r}]
runpy.run_path({str(entry)!r},run_name='import_fixture031')
sys.path.append({str(site)!r})
import torch
from scripts.z1_depth_campaign031.import_policy import verify_import_origins
manifest=json.loads(open({str(path)!r}).read())
before=(set(vars(torch.ops)),set(vars(torch.classes)))
verify_import_origins(manifest)
assert before==(set(vars(torch.ops)),set(vars(torch.classes)))
assert not any(n.split('.')[0] in ('src','mjlab','mujoco','mujoco_warp','rsl_rl') for n in sys.modules)
print('REAL_TORCH_IMPORT_ONLY031')
'''
    result=subprocess.run([sys.executable,'-B','-I','-S','-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='REAL_TORCH_IMPORT_ONLY031'


class OriginRejected031(ValueError):
    pass


def test_probe_origin_check_denies_science_after_counter_imports(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from scripts.z1_depth_campaign031 import probe
    events=[]
    fake_torch=types.ModuleType('torch');fake_torch.cuda=types.SimpleNamespace(is_available=lambda:True)
    train=types.ModuleType('scripts.train');train.run_train=lambda *a:events.append('model')
    factory=types.ModuleType('scripts.z1_depth_reward_training031');factory.build_train_config=lambda *a:None
    baseline=types.ModuleType('scripts.z1_split_credit_training021')
    for obj in (fake_torch,train,factory,baseline):monkeypatch.setitem(sys.modules,obj.__name__,obj)
    @contextmanager
    def counters(counts):
        events.append('counter-imports');yield []
    def reject():events.append('origin-check');raise OriginRejected031('stop before science')
    monkeypatch.setattr(probe,'counters',counters)
    monkeypatch.setattr(probe,'make_tiny',lambda *a:events.append('constructor'))
    with pytest.raises(OriginRejected031):probe.execute_probe(tmp_path,{'gates':[None,None,None,{'probe':probe.recipe()}]},before_science=reject)
    assert events==['counter-imports','origin-check']
    assert all(value==0 for key,value in files.read_json(tmp_path/'probe-actual-counts.json').items())


def test_evaluation_rechecks_loaded_factory_and_vendor_before_construction(tmp_path,monkeypatch):
    from scripts.z1_depth_campaign031 import evaluation as e
    events=[];checks=[]
    fake=types.ModuleType('torch');fake.cuda=types.SimpleNamespace(is_available=lambda:False)
    fake.set_num_threads=fake.set_num_interop_threads=lambda n:None
    fake.load=lambda *a,**kw:{}
    monkeypatch.setitem(sys.modules,'torch',fake)
    def check(manifest):
        checks.append(len(checks)+1);events.append('check')
        if len(checks)==2:raise OriginRejected031('stop before science')
    monkeypatch.setattr(worker,'verify_import_origins',check)
    monkeypatch.setattr(e,'verify_copies',lambda:{'pilot-compat001/qualified007/'+name:'a'*64 for name in ('horizon_integration.py','current_evaluator_candidate.py')})
    monkeypatch.setattr(e,'pilot_utilities',lambda:types.SimpleNamespace(adapt_pilot_condition=lambda *a:events.append('adapt')))
    monkeypatch.setattr(e,'measurement_context',lambda _: {})
    monkeypatch.setattr(e,'validate_payload',lambda *a:{})
    monkeypatch.setattr(e,'write_once',lambda *a:None)
    monkeypatch.setattr(e,'selected_configuration',lambda *a:events.append('factory'))
    def load(*args):
        events.append(args[0].name);return types.SimpleNamespace(HorizonAdapter=object,parse_args=lambda argv:types.SimpleNamespace())
    monkeypatch.setattr(e,'load_pinned',load)
    monkeypatch.setattr(e,'read_json',lambda _: {'sha256':'a'*64})
    monkeypatch.setattr(e,'sha_file',lambda _: 'a'*64)
    monkeypatch.setattr(e,'adapt_family_configuration',lambda *a:None)
    def denied_constructor(*args):
        events.append('constructor');raise AssertionError('science construction reached before origin check')
    monkeypatch.setattr(e,'make_adapter_class',lambda *a:denied_constructor)
    with monkeypatch.context() as patch:
        patch.setattr(sys,'path',list(sys.path))
        with pytest.raises(OriginRejected031):
            e.execute(dict(output=str(tmp_path),manifest={'source_root':str(tmp_path)},record={'checkpoint_path':'unused','task':'fixture031'}))
    assert events==['check','factory','horizon_integration.py','current_evaluator_candidate.py','reset_bank.py','check']


def test_source_namespace_rejects_conflicting_spec_search_path(import_tree,monkeypatch,tmp_path):
    from importlib.machinery import ModuleSpec
    manifest,modules,source,_=import_tree
    (source/'scripts').mkdir();path=source/'scripts/known031.py';path.write_text('VALUE=31\n')
    manifest['source_files']['scripts/known031.py']=files.sha_file(path)
    obj=types.ModuleType('scripts');obj.__path__=(str(source/'scripts'),)
    obj.__spec__=ModuleSpec('scripts',loader=None,is_package=True)
    obj.__spec__.submodule_search_locations=[str(tmp_path/'ambient/scripts')]
    modules['scripts']=obj
    with pytest.raises(ValueError):checked(manifest,modules,monkeypatch)


def test_originless_native_metadata_treatment_is_preserved(import_tree,monkeypatch):
    manifest,modules,_,_=import_tree
    for name in ('torch._C._dynamo.autograd_compiler','torch._VF','torch.backends.python_native'):
        modules[name]=types.ModuleType(name)
    checked(manifest,modules,monkeypatch)


def test_child_checks_runtime_path_before_compiler_or_native_initialization(import_tree,monkeypatch,tmp_path):
    from scripts.z1_depth_campaign031 import compiler
    manifest,_,_,site=import_tree;events=[]
    intent=dict(claim={'path':str(tmp_path/'gate-0.json')},parent={},nonce='n',private=str(tmp_path),output=str(tmp_path),kind='qualification',index=0)
    monkeypatch.setattr(worker,'read_json',lambda _:intent)
    monkeypatch.setattr(worker,'_validate_child',lambda *a:manifest)
    monkeypatch.setattr(worker,'write_once',lambda *a:None)
    monkeypatch.setattr(worker,'verify_package',lambda *a:events.append('source-runtime-verified'))
    monkeypatch.setattr(worker.life,'process_identity',lambda *a:{})
    monkeypatch.setattr(worker.os,'chdir',lambda *a:None)
    def denied_compiler(*args):
        events.append('compiler');raise AssertionError('compiler initialization reached before origin check')
    monkeypatch.setattr(compiler,'configure',denied_compiler)
    def reject(value):
        assert str(site) in sys.path
        events.append('origin-check');raise OriginRejected031('stop before native initialization')
    monkeypatch.setattr(worker,'verify_import_origins',reject)
    with monkeypatch.context() as patch:
        patch.setattr(sys,'path',list(sys.path))
        with pytest.raises(OriginRejected031):worker.execute_child('fixture-intent','digest')
    assert events==['source-runtime-verified','origin-check']
