"""Finite once-only eight-cell Slurm runner; imports no ML before admission.

One supervisor, one allocated GPU, a fresh process per cell. Failed or ambiguous
work is retained and stops the batch; this tool never submits or retries jobs.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import uuid

QUOTA_ROOT = '/cephhome/eunikhilr'
WORKER = 'scripts/z1_credit_critic_campaign.py'
LIMITS = dict(output_bytes=96*1024**2, cache_bytes=1024**3,
              quota_floor_bytes=512*1024**2, cell_wall_seconds=3600,
              total_wall_seconds=14400)
CRITICAL_SOURCE = {
    WORKER, 'scripts/train.py', 'scripts/z1_credit_critic_training.py',
    'src/tasks/hammer/config/z1/spline_credit_critic_cfg.py',
    'src/tasks/hammer/mdp/spline_credit_deadline.py',
    'src/tasks/hammer/rl/spline_credit_critic.py',
    'src/tasks/hammer/rl/spline_credit_critic_contract.py',
    'src/tasks/hammer/rl/credit_training_trace.py',
    'src/tasks/hammer/rl/impact_speed_contract.py', 'src/tasks/hammer/rl/runner.py',
    'src/tasks/hammer/rl/cat_ppo.py', 'src/tasks/hammer/rl/cat_storage.py',
    'src/tasks/hammer/cat/hook.py', 'src/tasks/hammer/mdp/sampled_pd500.py',
}
RUNTIME_SUFFIXES = dict(mjlab_runner='mjlab/rl/runner.py',
    rsl_runner='rsl_rl/runners/on_policy_runner.py', mjlab='mjlab/__init__.py',
    rsl_rl='rsl_rl/__init__.py', torch='torch/__init__.py',
    mujoco='mujoco/__init__.py', mujoco_warp='mujoco_warp/__init__.py')
RUNTIME_MODULES = dict(mjlab_runner='mjlab.rl.runner', rsl_runner='rsl_rl.runners.on_policy_runner',
                       mjlab='mjlab', rsl_rl='rsl_rl', torch='torch', mujoco='mujoco', mujoco_warp='mujoco_warp')
COMMON_INITIAL_HASHES = ('actor', 'shared_critic43', 'actor_normalizer',
                        'critic_normalizer', 'collection_torch_rng', 'collection_python_rng',
                        'collection_numpy_rng', 'collection_cuda_rng')
CACHE_KEYS = ('WARP_CACHE_PATH', 'CUDA_CACHE_PATH', 'TORCHINDUCTOR_CACHE_DIR',
              'TMPDIR', 'XDG_CACHE_HOME')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def mapping_digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def expected_cells():
    return [dict(index=i, credit_deadline_ms=(50,100,150,200)[i//2],
                 critic_mode=('current','event')[i%2], seed=2026102901,
                 updates=100, worlds=1024, rollout_steps=24) for i in range(8)]


def real(path):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts and path == path.resolve(),
         'canonical nonsymlink path required')
    for parent in path.parents:
        need(stat.S_ISDIR(parent.lstat().st_mode), 'real directory parents required')
    return path


def read_file(path, maximum=None):
    path = real(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode), 'regular file required')
        need(maximum is None or before.st_size <= maximum, 'file exceeds bound')
        body = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
    fields = ('st_dev','st_ino','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
    need(len(body) == before.st_size and all(getattr(before,k)==getattr(after,k) for k in fields),
         'file changed during read')
    return body


def file_sha(path):
    # External runtime libraries can be large; stream them without ML imports.
    path = real(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno()); need(stat.S_ISREG(before.st_mode), 'regular file required')
        h = hashlib.sha256(); count = 0
        while True:
            body = stream.read(1024**2)
            if not body: break
            count += len(body); need(count <= before.st_size, 'file grew during hash'); h.update(body)
        after = os.fstat(stream.fileno())
    fields = ('st_dev','st_ino','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
    need(count == before.st_size and all(getattr(before,k)==getattr(after,k) for k in fields), 'file changed during hash')
    return h.hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, 'duplicate JSON key'); result[key] = value
    return result


def load_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON value: '+value)
    return json.loads(read_file(path, 4*1024**2), object_pairs_hook=_pairs, parse_constant=invalid)


def write_once(path, value):
    path = real(path); body = encoded(value); need(len(body) <= 65536, 'record exceeds64KiB')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())
    need(read_file(path, 65536) == body, 'record readback differs')
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _verify_map(mapping, base=None):
    need(isinstance(mapping, dict) and mapping and len(mapping) <= 8192, 'complete nonempty file map required')
    for name, digest in mapping.items():
        need(isinstance(name,str) and re.fullmatch('[0-9a-f]{64}', digest or ''), 'valid file SHA required')
        relative = Path(name)
        if base is not None:
            need(not relative.is_absolute() and '..' not in relative.parts and str(relative)==name, 'source path escapes root')
        path = base / name if base is not None else Path(name)
        need(file_sha(path) == digest, 'pinned file bytes changed: '+name)


def admission_binding(manifest):
    return dict(campaign_root=manifest['campaign_root'], cells=manifest['cells'],
                worker_sha256=manifest['worker_sha256'], python_sha256=manifest['python_sha256'],
                source_files_sha256=mapping_digest(manifest['source_files']),
                runtime_files_sha256=mapping_digest(manifest['runtime_files']),
                asset_files_sha256=mapping_digest(manifest['asset_files']))


def verify_manifest(root, pin):
    root = real(root)
    need(root.parent == Path(QUOTA_ROOT)/'campaigns' and root.name.startswith('z1-credit-critic-screen-'), 'fresh campaign namespace required')
    need(re.fullmatch('[0-9a-f]{64}', pin or ''), 'manifest SHA required')
    need(file_sha(root/'manifest.json') == pin, 'bound manifest changed')
    m = load_json(root/'manifest.json')
    need(m.get('schema') == 'z1_credit_critic_screen_v1' and m.get('campaign_root') == str(root), 'manifest schema/root differs')
    need(encoded(m.get('cells')) == encoded(expected_cells()) and encoded(m.get('limits')) == encoded(LIMITS), 'finite cells or limits changed')
    source = real(root/'source'); declared = m.get('source_files', {})
    need(CRITICAL_SOURCE <= set(declared), 'critical source missing')
    files, directories = set(), set()
    for path in source.rglob('*'):
        need(not path.is_symlink() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'), 'source symlink/bytecode forbidden')
        mode = path.lstat().st_mode; name = str(path.relative_to(source))
        need(stat.S_ISREG(mode) or stat.S_ISDIR(mode), 'source special file forbidden')
        (directories if stat.S_ISDIR(mode) else files).add(name)
    allowed_dirs = {str(p) for name in declared for p in Path(name).parents if str(p) != '.'}
    need(files == set(declared) and directories == allowed_dirs, 'source member set differs')
    _verify_map(declared, source)
    need(real(__file__) == source/WORKER and file_sha(__file__) == m.get('worker_sha256'), 'running worker differs')
    executable = Path(m.get('python_executable',''))
    need(executable.is_absolute() and str(executable) == sys.executable and executable.resolve().is_file(), 'invoked Python differs')
    need(file_sha(executable.resolve()) == m.get('python_sha256'), 'resolved Python bytes changed')
    for key in ('runtime_files','asset_files'): _verify_map(m.get(key))
    roles = m.get('runtime_roles', {})
    need(set(roles) == set(RUNTIME_SUFFIXES), 'complete runtime roles required')
    for role, suffix in RUNTIME_SUFFIXES.items():
        need(roles[role].endswith('/'+suffix) and roles[role] in m['runtime_files'], 'runtime role differs')
    assets = m.get('asset_roles', {})
    need(set(assets) == {'scene_xml','robot_xml'}, 'complete XML asset roles required')
    need(all(path.endswith('.xml') and path in m['asset_files'] for path in assets.values()), 'asset role differs')
    for key in ('root_approval','admission'):
        ref = m.get(key)
        need(isinstance(ref,dict) and set(ref) == {'path','sha256'}, 'bound approval/admission missing')
        path = real(ref['path']); need(root in path.parents and file_sha(path) == ref['sha256'], 'admission reference changed')
        record = load_json(path)
        need(record.get('status') == 'PASS' and encoded(record.get('binding')) == encoded(admission_binding(m)), 'approval/admission not PASS for this package')
    need(m['root_approval']['path'] != m['admission']['path'], 'approval and admission must be separate records')
    return m


def allocation_identity():
    job, visible = os.environ.get('SLURM_JOB_ID',''), os.environ.get('CUDA_VISIBLE_DEVICES','')
    need(re.fullmatch('[1-9][0-9]*',job) and re.fullmatch('(?:[0-9]+|GPU-[0-9a-fA-F-]+)',visible), 'one exact Slurm GPU allocation required')
    r = subprocess.run(['nvidia-smi','-i',visible,'--query-gpu=uuid','--format=csv,noheader'], capture_output=True,text=True,timeout=10)
    need(r.returncode == 0 and len(r.stdout) <= 1024 and len(r.stderr) <= 1024, 'GPU identity query failed')
    uuid = r.stdout.strip()
    need(re.fullmatch('GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',uuid), 'single physical GPU UUID required')
    return dict(slurm_job_id=job, gpu_uuid=uuid, cuda_visible_devices=visible, hostname=os.uname().nodename)



def verify_import_origins(manifest, source, *, modules=None):
    modules = sys.modules if modules is None else modules
    for role, name in RUNTIME_MODULES.items():
        origin = getattr(modules.get(name), '__file__', None)
        need(origin is not None and Path(origin).resolve() == Path(manifest['runtime_roles'][role]).resolve(),
             'actual runtime import origin differs: '+name)
    for name, module in list(modules.items()):
        if name == 'src.tasks.hammer' or name.startswith('src.tasks.hammer.'):
            origin = getattr(module, '__file__', None)
            need(origin is not None and source in Path(origin).resolve().parents,
                 'actual source import escaped frozen package: '+name)
            need(str(Path(origin).resolve().relative_to(source)) in manifest['source_files'],
                 'actual source module is absent from manifest: '+name)


def quota_free():
    return int(os.getxattr(QUOTA_ROOT,'ceph.quota.max_bytes')) - int(os.getxattr(QUOTA_ROOT,'ceph.dir.rbytes'))


def directory_bytes(path):
    path = real(path)
    if not path.exists(): return 0
    need(path.is_dir(), 'directory required'); total = 0
    def walk_error(exc):
        if not isinstance(exc, FileNotFoundError): raise exc
    for base, directories, files in os.walk(path, followlinks=False, onerror=walk_error):
        for name in directories + files:
            try: info = (Path(base)/name).lstat()
            except FileNotFoundError: continue  # Cache writers may remove entries concurrently.
            need(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode), 'output/cache special file or symlink forbidden')
            if stat.S_ISREG(info.st_mode): total += info.st_size
    return total


def _cell(root, index):
    need(type(index) is int and 0 <= index < 8, 'cell index must be integer0..7')
    return root/'runs'/f'cell-{index:02d}', root/'private'/f'cell-{index:02d}'


def child_argv(root, pin, index):
    _cell(root,index)
    return [sys.executable,'-B','-I',str(root/'source'/WORKER),'--campaign',str(root),'--manifest-sha',pin,'--cell',str(index)]


def limit_reason(root, index, started, deadline, *, now=None):
    now = time.monotonic() if now is None else now
    if now >= deadline: return 'TOTAL_WALL_LIMIT'
    if now-started >= LIMITS['cell_wall_seconds']: return 'CELL_WALL_LIMIT'
    output, private = _cell(root,index)
    logs = root/'logs'; logbytes = sum((logs/f'cell-{index:02d}.{suffix}').stat().st_size for suffix in ('stdout','stderr') if (logs/f'cell-{index:02d}.{suffix}').exists())
    if directory_bytes(output)+logbytes > LIMITS['output_bytes']: return 'OUTPUT_LIMIT'
    if directory_bytes(private) > LIMITS['cache_bytes']: return 'CACHE_LIMIT'
    if quota_free() < LIMITS['quota_floor_bytes']: return 'QUOTA_FLOOR'
    return None


def validate_trace(row, update):
    def vector(value, width=2, *, low=None, high=None, boolean=False, integer=False):
        need(type(value) is list and len(value)==width, 'trace vector shape differs')
        for x in value:
            if boolean: need(type(x) is bool, 'trace boolean required')
            else:
                need(type(x) in ((int,) if integer else (int,float)) and math.isfinite(x), 'trace finite numeric required')
                need((low is None or x>=low) and (high is None or x<=high), 'trace numeric domain differs')
    def matrix(value, width, **kwargs):
        need(type(value) is list and len(value)==2, 'trace world matrix shape differs')
        for v in value: vector(v,width,**kwargs)
    need(type(row) is dict and row.get('schema')=='z1_credit_training_trace_v1'
         and type(row.get('update')) is int and row['update']==update
         and encoded(row.get('world_ids'))==b'[0,1]', 'trace update/scope differs')
    semantics=dict(scope='FIXED_TWO_WORLDS_NOT_POPULATION_STATISTICS',
        executed_action_semantics='ACTION_MANAGER_INPUT_AFTER_WRAPPER_CLIP_NOT_MOTOR_TORQUE',
        storage_reward_semantics='ACTUAL_ALREADY_CAT_SCALED_AND_TIMEOUT_BOOTSTRAPPED',
        raw_advantage_semantics='RETURNS_MINUS_VALUES_BEFORE_ADVANTAGE_NORMALIZATION')
    need(all(row.get(k)==v for k,v in semantics.items()), 'trace semantics missing or different')
    need(type(row.get('gamma')) in (int,float) and row['gamma']==0.99
         and type(row.get('lam')) in (int,float) and row['lam']==0.95, 'trace discount differs')
    vector(row.get('final_values'))
    steps=row.get('steps'); need(type(steps) is list and len(steps)==24, 'trace full rollout required')
    for step, record in enumerate(steps):
        need(type(record) is dict and type(record.get('step')) is int and record['step']==step, 'trace ordered step required')
        pre, stored=record.get('pre_reset'),record.get('stored')
        need(type(pre) is dict and type(stored) is dict, 'trace pre-reset and storage required')
        for name in ('raw_reward','accepted_credit_paid','r_neg_raw_minus_positive','r_pos','value_at_action'): vector(pre.get(name))
        vector(pre.get('delta'),low=0,high=1)
        vector(pre.get('contact_age_s'),low=0)
        vector(pre.get('unpaid_credit'),low=0,high=1)
        for name in ('native_reset','timeout','started'): vector(pre.get(name),boolean=True)
        need(pre.get('r_neg_semantics')=='RAW_MINUS_POSITIVE_NOT_UNROUNDED_PRODUCER_LEDGER', 'trace negative reward semantics differ')
        ledger=pre.get('reward_components')
        need(type(ledger) is dict and 0<len(ledger)<=64 and all(type(k) is str and k for k in ledger), 'trace reward ledger required')
        for value in ledger.values(): vector(value)
        need(ledger.get('accepted_spline_strike')==pre['accepted_credit_paid'], 'trace accepted reward join differs')
        for name in ('latent_actions','executed_actions','distribution_mean'): matrix(pre.get(name),13)
        matrix(pre.get('distribution_std'),13,low=0)
        need(all(x>0 for v in pre['distribution_std'] for x in v), 'trace distribution std must be positive')
        constraints=pre.get('constraints')
        need(type(constraints) is dict and pre.get('constraint_status')=='OBSERVED_LIVE_PRE_RESET', 'trace live constraints required')
        for name in ('delta_velocity','delta_impulse','delta'): vector(constraints.get(name),low=0,high=1)
        for name in ('delta_velocity_per_joint','delta_impulse_per_joint'): matrix(constraints.get(name),6,low=0,high=1)
        vector(constraints.get('winner_constraint'),integer=True,low=0,high=3)
        vector(constraints.get('winner_joint'),integer=True,low=-1,high=5)
        for name in ('lambda_per_joint','cap_utilization_per_joint','raw_margin_per_joint'): matrix(constraints.get(name),6)
        matrix(constraints.get('positive_margin_per_joint'),6,low=0)
        vector(constraints.get('active_limit_per_joint'),6,low=0)
        need(all(x>0 for x in constraints['active_limit_per_joint']), 'trace active limits must be positive')
        for name in ('values','rewards_after_cat_and_bootstrap','returns','next_values','raw_advantage','td_residual'): vector(stored.get(name))
        vector(stored.get('soft_dones'),low=0,high=1)
        vector(stored.get('hard_dones'),integer=True,low=0,high=1)
        need(stored['values']==pre['value_at_action'] and stored['soft_dones']==pre['delta']
             and stored['hard_dones']==pre['native_reset'], 'trace collected/storage join differs')


def verify_cuda_device(torch, identity):
    need(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.current_device()==0,
         'one available CUDA device at logical index0 required')
    value=getattr(torch.cuda.get_device_properties(0), 'uuid', None)
    if not isinstance(value, (str,bytes)):
        # PyTorch Module.cpp binds _CUuuid.bytes as vector<uint8_t> (a Python list).
        raw=getattr(value,'bytes',None)
        if type(raw) in (list,tuple):
            need(len(raw)==16 and all(type(x) is int and 0<=x<=255 for x in raw),
                 'actual CUDA UUID vector requires exactly16 uint8 integers')
            raw=bytes(raw)
        need(isinstance(raw,(bytes,bytearray,memoryview)) and len(raw)==16,
             'actual CUDA UUID object requires exactly16 raw bytes')
        value=bytes(raw)
    if isinstance(value, bytes):
        value=str(uuid.UUID(bytes=value)) if len(value)==16 else value.decode('ascii')
    need(type(value) is str, 'actual CUDA device UUID required')
    actual=uuid.UUID(value.removeprefix('GPU-'))
    need(actual==uuid.UUID(identity['gpu_uuid'].removeprefix('GPU-')), 'actual CUDA device differs from allocation UUID')
    return dict(logical_device=0,gpu_uuid='GPU-'+str(actual))


def final_artifacts(root, index):
    output, _ = _cell(root,index); initial = load_json(output/'initial-state.json')
    need(encoded(initial.get('cell')) == encoded(expected_cells()[index]) and type(initial.get('actual_counter')) is int and initial['actual_counter'] == 0 and initial.get('phase') == 'BEFORE_UPDATE_0_AND_FIRST_ROLLOUT', 'initial-state gate missing or different')
    hashes = initial.get('hashes', {})
    need(all(isinstance(hashes.get(k),str) and re.fullmatch('[0-9a-f]{64}',hashes[k]) for k in (*COMMON_INITIAL_HASHES,'critic')), 'initial fingerprints incomplete')
    observations = initial.get('initial_observations', {})
    need(all(isinstance(observations.get(k),str) and re.fullmatch('[0-9a-f]{64}',observations[k]) for k in ('actor_sha256','critic_shared43_sha256')), 'initial observation fingerprints incomplete')
    need(encoded(observations.get('actor_shape')) == b'[1024,43]' and encoded(observations.get('critic_shape')) == encoded([1024,47 if index%2 else 43]), 'initial observation shapes differ')
    checkpoints = {}
    for label in (0,24,49,99):
        path = output/'train'/f'model_{label}.pt'; digest = file_sha(path)
        record = load_json(output/f'checkpoint-{label}-live.json')
        need(type(record.get('iteration')) is int and record['iteration'] == label and type(record.get('common_step_counter')) is int and record['common_step_counter'] == (label+1)*24 and record.get('checkpoint_sha256') == digest, 'live checkpoint counter/digest differs')
        checkpoints[path.name] = digest
    trace = real(output/'training-trace.jsonl'); body = read_file(trace, 32*1024**2)
    lines = body.splitlines(); need(len(lines) == 100, 'complete100-update trace required')
    for update, line in enumerate(lines):
        need(len(line) <= 1024**2, 'trace line cap exceeded')
        row = json.loads(line, object_pairs_hook=_pairs, parse_constant=lambda x: need(False,'nonfinite trace'))
        validate_trace(row, update)
    return dict(updates=100, common_step_counter=2400, checkpoint_sha256=checkpoints,
                initial_state_sha256=file_sha(output/'initial-state.json'), initial_hashes=hashes,
                initial_observations=observations, trace_sha256=hashlib.sha256(body).hexdigest(), trace_bytes=len(body))


def _mkdir(path):
    path = real(path)
    if path.exists(): need(path.is_dir(), 'directory required')
    else:
        path.mkdir(mode=0o700)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)


def _kill(process):
    if process is None or process.poll() is not None: return
    os.killpg(process.pid,signal.SIGTERM)
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL); process.wait(timeout=10)


def supervise(root, pin):
    verify_manifest(root,pin); identity=allocation_identity(); root=Path(root)
    need(quota_free() >= LIMITS['quota_floor_bytes']+8*(LIMITS['output_bytes']+LIMITS['cache_bytes']), 'quota cannot reserve full eight-cell allowance')
    for name in ('claims','runs','private','logs'): _mkdir(root/name)
    for index in range(8):
        output,private=_cell(root,index)
        need(not os.path.lexists(output) and not os.path.lexists(private) and not os.path.lexists(root/'claims'/f'cell-{index:02d}.json'), 'cell path/claim already consumed')
    context=dict(schema='z1_credit_critic_supervisor_v1',supervisor_pid=os.getpid(),manifest_sha256=pin,allocation=identity)
    write_once(root/'claims/supervisor.json',context)
    completed=[]; global_deadline=time.monotonic()+LIMITS['total_wall_seconds']; reference=None
    for index in range(8):
        process=None; reason=None; started=time.monotonic(); code=None
        try:
            need(started < global_deadline, 'total wall allocation exhausted')
            need(quota_free() >= LIMITS['quota_floor_bytes']+LIMITS['output_bytes']+LIMITS['cache_bytes'], 'quota cannot reserve next cell')
            output,private=_cell(root,index); _mkdir(private)
            env=os.environ.copy()
            for key in CACHE_KEYS:
                directory=private/key.lower(); _mkdir(directory); env[key]=str(directory)
            env.update(LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',PYTHONDONTWRITEBYTECODE='1')
            with (root/'logs'/f'cell-{index:02d}.stdout').open('xb') as stdout, (root/'logs'/f'cell-{index:02d}.stderr').open('xb') as stderr:
                process=subprocess.Popen(child_argv(root,pin,index),stdout=stdout,stderr=stderr,env=env,start_new_session=True)
                while process.poll() is None:
                    reason=limit_reason(root,index,started,global_deadline)
                    if reason: _kill(process); break
                    time.sleep(1)
                code=process.returncode
            reason = reason or limit_reason(root,index,started,global_deadline)
            need(reason is None and code == 0, 'child failed or finite monitor stopped it')
            artifact=final_artifacts(root,index); terminal=load_json(output/'train-complete.json')
            need(terminal.get('status') == 'COMPLETE' and terminal.get('manifest_sha256') == pin and terminal.get('allocation') == identity and terminal.get('artifacts') == artifact, 'child completion not independently joined')
            common={k:artifact['initial_hashes'][k] for k in COMMON_INITIAL_HASHES}
            common.update({k:artifact['initial_observations'][k] for k in ('actor_sha256','critic_shared43_sha256')})
            if reference is None: reference=common
            need(common == reference, 'paired initialization differs across cells')
            completed.append(index)
        except BaseException as exc:
            _kill(process)
            reason=reason or f'{type(exc).__name__}: {str(exc)[:2048]}'
        result=dict(status='COMPLETE' if reason is None else 'FAILED_NO_RETRY',cell=index,
                    returncode=code,reason=reason,allocation=identity,automatic_retry=False,
                    started_monotonic=started,finished_epoch=time.time())
        write_once(root/'logs'/f'cell-{index:02d}-terminal.json',result)
        if reason is not None: break
    result=dict(status='COMPLETE' if len(completed)==8 else 'FAILED_NO_RETRY',completed_cells=completed,
                failed_cell=None if len(completed)==8 else index,
                remaining_cells_not_attempted=[] if len(completed)==8 else list(range(index+1,8)),
                allocation=identity,manifest_sha256=pin,automatic_retry=False,finished_epoch=time.time())
    write_once(root/'supervisor-result.json',result)
    return result


def run_cell(root, pin, index):
    manifest=verify_manifest(root,pin); identity=allocation_identity(); root=Path(root); output,private=_cell(root,index)
    context=load_json(root/'claims/supervisor.json'); parent=os.getppid()
    need(context.get('supervisor_pid') == parent and context.get('manifest_sha256') == pin and context.get('allocation') == identity, 'child requires same live parent/job/GPU claim')
    expected=child_argv(root,pin,index)[:-2]+['--supervise']
    need(sys.platform == 'linux' and Path(f'/proc/{parent}/cmdline').read_bytes().split(b'\0')[:-1] == [x.encode() for x in expected], 'supervisor command identity differs')
    need(ctypes.CDLL(None).prctl(1,signal.SIGTERM)==0 and os.getppid()==parent, 'cannot bind supervisor death')
    need(not os.path.lexists(output) and private.is_dir() and directory_bytes(private)==0, 'fresh output and empty private cache required')
    need(quota_free() >= LIMITS['quota_floor_bytes']+LIMITS['output_bytes'], 'preservation quota unavailable')
    write_once(root/'claims'/f'cell-{index:02d}.json',dict(cell=expected_cells()[index],parent_pid=parent,allocation=identity,manifest_sha256=pin))
    _mkdir(output)
    try:
        for key in CACHE_KEYS:
            path=real(private/key.lower()); need(path.is_dir() and os.environ.get(key)==str(path), 'private cache binding differs')
        sys.dont_write_bytecode=True; os.environ['PYTHONDONTWRITEBYTECODE']='1'; os.chdir(private); sys.path.insert(0,str(root/'source'))
        import torch
        write_once(output/'gpu-runtime.json',verify_cuda_device(torch,identity))
        import src.tasks  # noqa: F401 — admitted package registration only.
        from scripts.train import run_train
        from scripts.z1_credit_critic_training import build_train_config, selected_checkpoint_saves, matched_initial_state_gate, cell_for_index
        need(encoded(cell_for_index(index)) == encoded(manifest['cells'][index]), 'adapter cell differs')
        for name in ('scripts.train','scripts.z1_credit_critic_training'):
            need(Path(sys.modules[name].__file__).resolve() == root/'source'/Path(*name.split('.')).with_suffix('.py'), 'training import escaped frozen source')
        verify_import_origins(manifest,root/'source')
        task,cfg=build_train_config(index)
        reference = load_json(root/'runs/cell-00/initial-state.json') if index > 0 else None
        with selected_checkpoint_saves(index,output), matched_initial_state_gate(index,output,enable_trace=True,reference_initial_state=reference):
            run_train(task,cfg,output/'train')
        result=dict(status='COMPLETE',cell=index,manifest_sha256=pin,allocation=identity,artifacts=final_artifacts(root,index),automatic_retry=False,finished_epoch=time.time())
        write_once(output/'train-complete.json',result)
        return result
    except BaseException as exc:
        write_once(output/'failure.json',dict(status='FAILED_NO_RETRY',error_type=type(exc).__name__,error=str(exc)[:4096],manifest_sha256=pin,automatic_retry=False,finished_epoch=time.time()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--manifest-sha',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cell',type=int); mode.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode, 'run worker with exact Python -B -I flags')
    result=supervise(args.campaign,args.manifest_sha) if args.supervise else run_cell(args.campaign,args.manifest_sha,args.cell)
    print(encoded(result).decode())
    return 0 if result['status']=='COMPLETE' else 1


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(encoded(dict(status='FAILED_NO_RETRY',error_type=type(exc).__name__,error=str(exc)[:4096])).decode(),file=sys.stderr)
        raise SystemExit(1)
