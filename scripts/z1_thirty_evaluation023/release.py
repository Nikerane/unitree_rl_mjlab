"""Closed sixty-case scope; prospective hashes remain unresolved until sealing."""
import math
from pathlib import Path
import re
from .files import canonical, digest, need, read_json, sha_file

ROOT = '/cephhome/eunikhilr/campaigns/z1-thirty-evaluation-20261031-v1'
PRODUCER_ROOT = '/cephhome/eunikhilr/campaigns/z1-training-thirty-20261030-v2'
PRODUCER_SHA = '77a3e31704964e681c775d2e5c37c9ea4c15ca8d3291f639ce8cf56a478be9b1'
BANK_SHA = 'b0baebc9dee774c89091a16ed7ba6ce5bfb3df4e4a639b6faabf56ec9da43f22'
BANK_ADAPTER_SHA = '63cb613f85f7e3121690fd15b56eb3c3816ad3ac03c154e0385cb9010ab44379'
EVALUATION_SEED = 2026103001
DEFERRED = 'DEFERRED_MUST_RESOLVE_FROM_GENUINE_COMPLETED_PRODUCER_SEAL_AND_VERIFY_FILE_BEFORE_LOAD'
MIB = 1024**2
LIMITS = dict(total_bytes=3096*MIB, quota_floor=1024**3, raw_bytes=16*MIB,
              private_bytes=32*MIB, package_bytes=16*MIB, archive_bytes=8*MIB,
              control_bytes=64*MIB, reserve_bytes=128*MIB, stdout_bytes=MIB,
              stderr_bytes=MIB, case_seconds=7200, batch_seconds=8100, cleanup_seconds=60)
PRODUCER_JOBS = ['48951832','48951836','48951837','48951838','48951839','48951840','48951842','48951843','48951844']
MATRIX_SHA = '5c032085c5dd58abc86e7b3e7f9a5437d43bd14520c613c57e5f764e03ea349c'


def expected_cases():
    path = Path(__file__).with_name('matrix.json')
    need(sha_file(path) == MATRIX_SHA, 'prospective matrix bytes differ')
    return read_json(path)


def case_for_index(index):
    need(type(index) is int and 0 <= index < 60, 'case outside finite sixty')
    return expected_cases()[index]


def package_map(files):
    need(type(files) is dict and files, 'package inventory required')
    for name, pin in files.items():
        p = Path(name)
        need(type(name) is str and not p.is_absolute() and '..' not in p.parts and str(p) == name,
             'package member path differs')
        need(type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin), 'package SHA256 required')
    return dict(files)


def build_release(package_files):
    """Proposal constructor only. Root separately signs its canonical digest."""
    return dict(schema='z1_thirty_evaluation_release023', owner='root', scope='evaluation60_bank32',
                root=ROOT, package_root=ROOT+'/package/z1_thirty_evaluation023',
                producer_manifest=dict(path=PRODUCER_ROOT+'/manifest.json', sha256=PRODUCER_SHA),
                source_root=PRODUCER_ROOT+'/source', producer_jobs=PRODUCER_JOBS,
                package_files=package_map(package_files), cases=expected_cases(), limits=LIMITS,
                reset_bank_sha256=BANK_SHA, bank_adapter_sha256=BANK_ADAPTER_SHA,
                evaluation_seed=EVALUATION_SEED, contexts=32, case_count=60, nominal_bank_starts=1920,
                allocation=dict(cpus=2,memory_mib=8192,threads=2,gpus=0,max_active_array=7),
                canary_case=0, array_indices=list(range(1,60)), retry=False, resume=False, requeue=False,
                physical_warmup_starts=0, automatic_cache_deletion=False)


def validate_release(release):
    need(type(release) is dict, 'evaluation release required')
    need(canonical(release) == canonical(build_release(release.get('package_files'))), 'closed evaluation release differs')
    return release


def validate_grant(release, grant, *, now):
    validate_release(release)
    need(type(grant) is dict and set(grant) == {'schema','owner','scope','release_sha256','issued_epoch','expires_epoch'}, 'exact separate evaluation grant required')
    need(grant['schema'] == 'z1_thirty_evaluation_grant023' and grant['owner'] == 'root'
         and grant['scope'] == 'evaluation60_bank32', 'root evaluation authority required')
    need(grant['release_sha256'] == digest(release), 'grant release digest differs')
    issued, expiry = grant['issued_epoch'], grant['expires_epoch']
    need(all(type(v) in (int,float) and math.isfinite(v) for v in (issued, expiry, now)), 'invalid authority clock')
    need(0 <= issued <= now < expiry and 0 < expiry-issued <= 72*3600, 'expired/future/overlong evaluation grant')
    return grant


def validate_cpu_allocation(environment, hostname, affinity):
    need(type(hostname) is str and re.fullmatch(r'cn[0-9]{4}',hostname), 'CPU host required')
    need(environment.get('SLURM_CPUS_PER_TASK') == '2' and environment.get('SLURM_MEM_PER_NODE') == '8192'
         and len(affinity) == 2, 'CPU/memory/affinity differs')
    need(environment.get('CUDA_VISIBLE_DEVICES') == '' and environment.get('SLURM_GPUS_ON_NODE','0') in ('','0')
         and environment.get('SLURM_GPUS','0') in ('','0'), 'GPU must be hidden and unallocated')
    need(all(environment.get(k) == '2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')), 'thread cap differs')
    need(re.fullmatch('[1-9][0-9]*', environment.get('SLURM_JOB_ID','')), 'Slurm job identity required')
