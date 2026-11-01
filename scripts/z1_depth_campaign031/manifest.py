"""Closed matrix, resource envelope and actual-workspace source freeze."""
from pathlib import Path
import ast
import os
import shutil
from .files import canonical, digest, need, read_json, reference, relative_name, safe_path, sha_file, sync_dir, write_once

MIB = 1024**2
GIB = 1024**3
ROOT = '/cephhome/eunikhilr/campaigns/z1-depth-reward-20261101-v3'
EVAL_ROOT = '/cephhome/eunikhilr/campaigns/z1-depth-evaluation-20261101-v3'
BACKUP_ROOT = '/Volumes/NikeraneSSD/Z1-campaign-backups/2026-11-01-depth-reward031'
SLURM_ACCOUNT = 'd2026d06-166-users'
SLURM_QOS = 'normal'
SSD_UUID = 'E345FED4-EE51-36D0-AB62-A4392E745189'
SEEDS = (2026110101, 2026110102, 2026110103)
CAP_BYTES = (12*256+12*128+24*16+24*32+768+768+256)*MIB
LIMITS = dict(training_output=256*MIB, trace=160*MIB, population=16*MIB, logs=32*MIB,
    checkpoints=8*MIB, other=40*MIB, training_private=128*MIB, eval_output=16*MIB,
    eval_private=32*MIB, qualification=768*MIB, source_assets_controls=768*MIB,
    contingency=256*MIB, cap=CAP_BYTES, free_floor=GIB, external_growth=2*GIB,
    gpu_concurrency=6, eval_concurrency=7, policy_seconds=7200, slot_seconds=16200,
    slot_required_seconds=16030, anchor_wait_seconds=600, global_gpu_seconds=28800, scheduler_exit_margin_seconds=180,
    qualification_seconds=2880, eval_seconds=7200, eval_allocation_seconds=8100,
    eval_cpus=2, eval_memory_gib=8, controller_seconds=64800, controller_cpus=1,
    controller_memory_gib=2, maximum_case_cpu_hours=108, maximum_controller_cpu_hours=18)


def training_cell(index):
    need(type(index) is int and 0<=index<12, 'integer cell0..11 required')
    return dict(index=index, arm=('A0','A1','A2','A3')[index%4], seed=SEEDS[index//4],
        setting='vic_baseline', critic_mode='current', updates=500, worlds=1024,
        rollout_steps=24, early_fraction=.5, early_deadline_ms=50, reserve_deadline_ms=200)


def template():
    from .admission import origin_spec
    cells = [dict(training_cell(i), slot=2*(i//4)+(i%4)//2, reference_index=4*(i//4),
        output=f'{ROOT}/outputs/cell-{i:02d}', private=f'{ROOT}/private/cell-{i:02d}') for i in range(12)]
    return dict(schema='z1_depth_campaign031', status='PREPARED_UNISSUED', root=ROOT,
        evaluation_root=EVAL_ROOT, backup_root=BACKUP_ROOT, ssd_uuid=SSD_UUID,
        source_root=ROOT+'/source', cells=cells, admission_recovery=origin_spec(),
        slots=[dict(index=s,cells=[4*(s//2)+2*(s%2),4*(s//2)+2*(s%2)+1]) for s in range(6)],
        cases=[dict(index=i,cell=i//2,label=(99,499)[i%2],counter=(2400,12000)[i%2],
                    output=f'{EVAL_ROOT}/outputs/case-{i:02d}',private=f'{EVAL_ROOT}/private/case-{i:02d}') for i in range(24)],
        limits=dict(LIMITS),slurm=dict(account=SLURM_ACCOUNT,qos=SLURM_QOS,gpu_partition='gpu',cpu_partition='cpu'), resume=False,retry=False,requeue=False)


def validate(value):
    base = template()
    need(type(value) is dict, 'manifest dictionary required')
    for key, expected in base.items():
        need(canonical(value.get(key))==canonical(expected), 'closed manifest differs: '+key)
    need(set(value)<=set(base)|{'source_files','runtime_files','asset_files','compiler_policy','python',
        'seed_history','qualification_recipe','vendor_files','provenance'}, 'unknown manifest fields')
    if 'qualification_recipe' in value:
        need(canonical(value['qualification_recipe'])==canonical(qualification_recipe()),'closed qualification recipe differs')
    return value


def qualification_recipe():
    specs = [('validate_rewards',300,2,10,97,'ALL PHASES PASSED'),
             ('verify_contact_sensor',300,4,4,800,'ContactSensor verified.'),
             ('verify_reward_setup',600,16,48,6480,'Reward setup verified. Safe to start training.')]
    gates=[dict(name=n,seconds=t,constructor_worlds=c,reset_worlds=r,world_transitions=w,
                learned_forwards=0,updates=0,script='docs/research/reward-design/'+n+'.py',argv=[],pass_marker=p)
           for n,t,c,r,w,p in specs]
    from .probe import recipe as probe_recipe
    gates.append(dict(probe=probe_recipe(),name='depth031_probe',seconds=720,production=dict(arm='A3',worlds=1024,
        updates=1,policy_steps=24,world_transitions=24576,constructor_worlds=1024),
        script='scripts/z1_depth_campaign031/probe.py',argv=[]))
    return dict(schema='z1_depth_qualification_recipe031',gates=gates,per_gate_overhead_seconds=70,
        publication_cleanup_exit_seconds=270,total_bound_seconds=2470,allocation_seconds=2640,
        native_autoresets='observed separately; bounded by physical transitions',retry=False)


def copy_sources(source, destination, names):
    source=safe_path(source);destination=safe_path(destination)
    destination.mkdir(exist_ok=False)
    pins={}
    for name in sorted(set(names)):
        rel=relative_name(name);original=safe_path(source/rel)
        before=reference(original);target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(original.read_bytes());stream.flush();os.fsync(stream.fileno())
        need(sha_file(target)==before['sha256'] and reference(original)==before, 'source changed while freezing')
        pins[name]=before['sha256']
    sync_dir(destination)
    return pins


def native_asset_root(source_root):
    """The unchanged nail_block resolver reaches the campaign's sibling tree."""
    module=Path(source_root)/'src/tasks/hammer/nail_block.py'
    return module.parents[4]/'safe_impact_manipulation/hammer_z1_env/assets'


def asset_copy_plan(rows,source_root):
    root=native_asset_root(source_root)
    return dict(authority=False,files=[dict(source=row['path'],
        destination=str(root/relative_name(row['path'].split('/assets/',1)[1])),sha256=row['sha256']) for row in rows])


def verify_asset_layout(manifest,*,source_tree=None):
    """Authenticate and inspect the resolver AST without importing native code."""
    relative='src/tasks/hammer/nail_block.py'
    native=Path(manifest['source_root'] if source_tree is None else source_tree)/relative
    need(sha_file(native)==manifest['source_files'][relative],'pinned asset resolver bytes differ')
    tree=ast.parse(native.read_bytes())
    values=[node.value for node in tree.body if isinstance(node,ast.AnnAssign)
            and isinstance(node.target,ast.Name) and node.target.id=='_SCENE_XML']
    expected=ast.parse("Path(__file__).resolve().parents[4] / 'safe_impact_manipulation' / 'hammer_z1_env' / 'assets' / 'nail_block_scene.xml'",mode='eval').body
    need(len(values)==1 and ast.dump(values[0],include_attributes=False)==ast.dump(expected,include_attributes=False),
         'native asset resolver expression changed; independent review required')
    root=native_asset_root(manifest['source_root'])
    need(root==Path(manifest['root'])/'safe_impact_manipulation/hammer_z1_env/assets','asset root differs from native resolver')
    need(all(Path(path).is_relative_to(root) and Path(path)!=root for path in manifest['asset_files'])
         and str(root/'nail_block_scene.xml') in manifest['asset_files'],'asset pins do not cover the native resolver layout')
    return root


def runtime_files_with_virtual_owners(runtime,python):
    """Add only the two installed Torch implementations omitted by baseline375."""
    from .import_policy import runtime_site
    site=runtime_site(dict(runtime_files=runtime,python=python))
    extra={str(site/'torch/_ops.py'):'c3d561d13aa4dede4525a5df8b47e38a54135653d1907101211c843df467a519',
        str(site/'torch/_classes.py'):'0db3dc302e28fd085065ea2883ce33fa4af406b9fd1439f2a431f053e9a764b9'}
    need(all(path not in runtime or runtime[path]==sha for path,sha in extra.items()),'virtual runtime pins differ')
    return {**runtime,**extra}


def prepare(workspace, destination, evidence):
    """Local unissued package. Never creates the remote/SSD campaign namespaces."""
    workspace=safe_path(workspace);evidence=safe_path(evidence);destination=safe_path(destination)
    need(not destination.exists(), 'prepared destination consumed')
    identities=read_json(evidence/'runtime-assets-current-identities031.json')
    hooks=read_json(evidence/'task-2-source-pins.json')
    required={**hooks['owned_files'],
        'src/tasks/hammer/rl/impact_speed_contract.py':hooks['after_impact_speed_contract_sha256'],
        'scripts/z1_critic_noise_training021.py':hooks['after_z1_critic_noise_training021_sha256'],
        'src/tasks/hammer/rl/split_credit_training_trace.py':hooks['after_split_credit_training_trace_fix1_sha256']}
    for name,sha in required.items():
        need(sha_file(workspace/name)==sha, 'reviewed Task2 bytes changed: '+name)
    groups=identities['remote']['groups']
    need(groups['runtime_files']['count']==375 and groups['asset_files']['count']==32 and not identities['remote']['mismatches'],
         'refreshed runtime/assets baseline differs')
    # The old object supplies a reviewed code-content inventory only. No old
    # grant, release status, registry, claim or qualification authority is copied.
    baseline=identities['baseline_only']
    need(sha_file(baseline['path'])==baseline['sha256'],'content-baseline reference changed')
    content=read_json(baseline['path'])['source_files']
    need(len(content)==688,'reviewed source-content baseline count differs')
    for name,sha in content.items():
        need(sha_file(workspace/relative_name(name))==required.get(name,sha),'unreviewed source delta: '+name)
    names=list(content)
    names.extend(required)
    names.extend(('src/tasks/hammer/mdp/depth_credit031.py','tests/test_depth_credit031.py','tests/test_depth_campaign031.py','tests/test_depth_join031.py','tests/test_depth_admission031.py','tests/test_depth_import_ownership031.py'))
    for p in (workspace/'scripts/z1_depth_campaign031').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:names.append(str(p.relative_to(workspace)))
    from .admission import ORIGIN_REF, SECOND_REF, validate_origin_record, validate_second_record
    origin=read_json(evidence/'first-gpu-allocation-accounting031.json')
    need(origin['reference']==ORIGIN_REF,'first allocation reference differs')
    validate_origin_record(origin['record'])
    second=read_json(evidence/'second-technical-allocation-accounting031.json')
    need(second['reference']==SECOND_REF,'second allocation reference differs')
    need((evidence/'namespace-diagnosis031/second-technical-allocation-accounting-body031.json').read_bytes()==canonical(second['record'])+b'\n',
         'second canonical body differs')
    validate_second_record(second['record'])
    destination.mkdir()
    pins=copy_sources(workspace,destination/'source',names)
    manifest=template();manifest['source_files']=pins
    manifest['runtime_files']=runtime_files_with_virtual_owners(
        {r['path']:r['sha256'] for r in groups['runtime_files']['files']},identities['remote']['python'])
    copy_plan=asset_copy_plan(groups['asset_files']['files'],manifest['source_root'])
    manifest['asset_files']={row['destination']:row['sha256'] for row in copy_plan['files']}
    verify_asset_layout(manifest,source_tree=destination/'source')
    from .compiler import policy
    compiler_observation=read_json(evidence/'compiler-content-preflight031.json')
    need(compiler_observation['all_match'] is True and {r['path']:r['sha256'] for r in compiler_observation['files']}==policy()['runtime_files'],'refreshed compiler bytes differ')
    manifest['compiler_policy']=policy();manifest['python']=identities['remote']['python']
    manifest['qualification_recipe']=qualification_recipe()
    manifest['seed_history']=dict(scope='same unused technical-release labels; fresh claims/checkpoints proof required before admission',
        seeds=list(SEEDS),evidence=reference(evidence/'ops-task3-interface.md'))
    manifest['provenance']=dict(reviewed_task2=reference(evidence/'task-2-source-pins.json'),
        runtime_observation=reference(evidence/'runtime-assets-current-identities031.json'),
        compiler_observation=reference(evidence/'compiler-content-preflight031.json'),
        first_allocation=dict(ORIGIN_REF),second_allocation=dict(SECOND_REF),workspace_bytes=True,authority=False)
    manifest['vendor_files']=read_json(workspace/'scripts/z1_depth_campaign031/COPIED-SOURCES031.json')
    validate(manifest)
    write_once(destination/'manifest.json',manifest)
    write_once(destination/'qualification-recipe.json',manifest['qualification_recipe'])
    write_once(destination/'asset-copy-plan.json',copy_plan)
    return manifest


def verify_package(manifest):
    validate(manifest)
    from .admission import authenticate_origin, ORIGIN_REF, SECOND_REF
    need(manifest['provenance']['first_allocation']==ORIGIN_REF
         and manifest['provenance']['second_allocation']==SECOND_REF,'two-failure provenance differs')
    authenticate_origin()
    need(len(manifest['runtime_files'])==377 and len(manifest['asset_files'])==32,'exact runtime/assets closure required')
    need(manifest['runtime_files']==runtime_files_with_virtual_owners(manifest['runtime_files'],manifest['python']),
         'virtual runtime closure differs')
    source=Path(manifest['source_root'])
    asset_root=verify_asset_layout(manifest)
    expected=set(manifest['source_files'])
    actual=set()
    for path in source.rglob('*'):
        safe_path(path)
        if path.is_file():actual.add(str(path.relative_to(source)))
    need(actual==expected,'source closure has missing or injected members')
    asset_members=set()
    for path in asset_root.rglob('*'):
        safe_path(path)
        if not path.is_dir():
            need(path.is_file(),'nonregular asset member refused')
            asset_members.add(str(path))
    need(asset_members==set(manifest['asset_files']),'asset closure has missing or injected members')
    for rel,sha in manifest['source_files'].items():
        need(sha_file(Path(manifest['source_root'])/relative_name(rel))==sha,'frozen source differs: '+rel)
    for key in ('runtime_files','asset_files'):
        for path,sha in manifest[key].items():need(sha_file(path)==sha,'frozen '+key+' differs: '+path)
    p=manifest['python'];need(str(Path(p['path']).resolve())==p['resolved'] and sha_file(p['resolved'])==p['sha256'], 'Python binary differs')
    for path,sha in manifest['compiler_policy']['runtime_files'].items():need(sha_file(path)==sha,'compiler bytes differ')
    return manifest
