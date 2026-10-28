#!/usr/bin/env python3
"""CPU-only terminal archive with separate Mac and Vega future reserves.

Preparation only. This program is not run as part of its implementation.
It creates no Slurm work, seed claims, remote files, or deletions.
"""
from __future__ import annotations
from archive_contract import validate_paths, validate_layout, SHARED_SHA, CAMPAIGN, validate_identity_capture

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import threading
import time

FLOOR = 512 * 1024**2
CONTROL_RESERVE = 32 * 1024**2
MAX_WALL_SECONDS = 600
SCOPES = {"training", "own_cpu32", "science160"}
REMOTE_FUTURE_MIN_BYTES = 4563402752


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_deadline(deadline):
    require(time.monotonic() < deadline, "terminal multiroot archive wall deadline expired")


def sha_file(path, deadline=None):
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if deadline is not None:
                check_deadline(deadline)
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def save_exclusive(path, data, *, reserve=CONTROL_RESERVE, deadline=None):
    path = Path(path)
    check_local_floor(path.parent, len(data), reserve)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        while view:
            if deadline is not None:
                check_deadline(deadline)
            check_local_floor(path.parent, min(len(view), 1024 * 1024), reserve)
            written = os.write(fd, view[:1024 * 1024])
            require(written > 0, "short local evidence write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def check_local_floor(path, next_bytes=0, reserve=0):
    free = shutil.disk_usage(path).free
    require(free - next_bytes - reserve >= FLOOR,
            "local archive write would cross the 512MiB preservation floor")
    return free


def regular_bound(spec, base, maximum, deadline):
    require(isinstance(spec, dict) and set(spec) == {"path", "sha256"},
            "terminal evidence requires an exact path/SHA-256 binding")
    path = Path(spec["path"])
    path = path if path.is_absolute() else base / path
    path = Path(os.path.abspath(path))
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_size <= maximum,
            "terminal evidence must be a bounded regular file")
    digest = sha_file(path, deadline)
    require(digest == spec["sha256"], "terminal evidence hash differs")
    return path, digest, info.st_size


def read_local_json(path, maximum, deadline):
    info = Path(path).lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_size <= maximum,
            "bound root evidence must be a bounded regular nonsymlink file")
    chunks, total = [], 0
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        while True:
            check_deadline(deadline)
            block = stream.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            require(total <= maximum, "root evidence exceeded metadata cap while reading")
            chunks.append(block)
    return json.loads(b"".join(chunks))


def copy_bound_file(source_path, target_path, expected_sha, expected_bytes, deadline, reserve=CONTROL_RESERVE):
    source_fd = os.open(source_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(source_fd, "rb") as source:
        fd = os.open(target_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest, size = hashlib.sha256(), 0
        try:
            while True:
                check_deadline(deadline)
                block = source.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                require(size <= expected_bytes, "bound local evidence grew during copy")
                check_local_floor(Path(target_path).parent,len(block),reserve)
                digest.update(block)
                view = memoryview(block)
                while view:
                    written = os.write(fd, view)
                    require(written > 0, "short bound evidence copy")
                    view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
    require(size == expected_bytes and digest.hexdigest() == expected_sha,
            "bound evidence copy differs from its exact source hash/size")


def load_bound(spec, base, maximum, deadline):
    path,digest,size=regular_bound(spec,base,maximum,deadline)
    return read_local_json(path,maximum,deadline),dict(path=str(path),sha256=digest,bytes=size)


def validate(request, request_path, deadline):
    require(request.get('schema')=='z1_horizon_terminal_multiroot_archive_request_v1','new explicit archive request required')
    require(request.get('scope')in('own_cpu32','science160'),'CPU-only capacity-split helper; training admission is outside this version')
    for key in ('max_source_bytes','max_archive_bytes','max_members','max_metadata_bytes','max_evidence_bytes','wall_seconds','future_reserve_bytes','remote_future_reserve_bytes'):
        require(type(request.get(key))is int and request[key]>0,'finite positive '+key+' required')
    require(request['remote_future_reserve_bytes']>=REMOTE_FUTURE_MIN_BYTES,'remote future reserve must retain at least4563402752B')
    require(request['max_members']<=100000 and request['max_metadata_bytes']<=16*1024**2 and request['max_evidence_bytes']<=16*1024**2 and request['wall_seconds']<=600,'hard bound exceeded')
    require(request.get('ssh_host')=='vega' and Path(request.get('remote_python','')).is_absolute(),'explicit Vega interpreter required')
    require(request.get('quota_root')=='/cephhome/eunikhilr','exact quota root required')
    require(sys.platform=='darwin','independent local Mac readback required')
    if request['scope']!='training':
        layout,layout_bound=load_bound(request.get('cpu_layout_reference'),request_path.parent,request['max_evidence_bytes'],deadline)
        require(request.get('cpu_layout')==layout,'inline CPU layout must equal bound prelaunch layout')
        validate_layout(layout)
    roots=validate_paths(request)
    request['_bound_extra']=[]
    identity,identity_bound=load_bound(request.get('scheduler_identity_reference'),request_path.parent,request['max_evidence_bytes'],deadline)
    source,source_bound=load_bound(identity.get('source_reference'),request_path.parent,request['max_evidence_bytes'],deadline)
    require(request.get('scheduler_identity_capture')==identity and identity.get('job_id')==request['job_id'],'inline scheduler identity differs from bound actual capture')
    validate_identity_capture(identity,source)
    request['_bound_extra'].extend([identity_bound,source_bound])
    if request['scope']!='training':
        request['_bound_extra'].append(layout_bound)
        cpu_manifest,manifest_bound=load_bound(request.get('cpu_supervisor_manifest'),request_path.parent,request['max_evidence_bytes'],deadline)
        cpu_release,release_bound=load_bound(request.get('cpu_supervisor_release'),request_path.parent,request['max_evidence_bytes'],deadline)
        request['_bound_extra'].extend([manifest_bound,release_bound])
        require(cpu_release.get('status')=='ADMITTED_ONCE' and cpu_release.get('manifest',{}).get('sha256')==manifest_bound['sha256'] and cpu_manifest.get('schema')=='z1_horizon_cpu_supervisor_v1','actual CPU release/manifest binding required')
        expected={(row['policy_id'],row['scope'],row['bank']):(row['run_root'],row['private_root']) for row in layout['slots']}
        actual={}
        for slot in cpu_manifest.get('slots',[]):
            intent=slot['intent'];key=(intent['policy']['policy_id'],intent['scope'],None if intent['scope']=='own_cpu32' else intent['evaluation_seed'])
            require(key not in actual,'duplicate supervisor CPU mapping')
            actual[key]=(intent['output_root'],slot['private_root'])
        require(actual==expected,'closed CPU archive layout differs from actual prelaunch supervisor slots')
        selected=next(slot for slot in cpu_manifest['slots']if(
            slot['intent']['policy']['policy_id'],slot['intent']['scope'],
            None if slot['intent']['scope']=='own_cpu32'else slot['intent']['evaluation_seed'])==(
                request['policy_id'],request['scope'],request.get('bank')))
        selected_future=selected.get('limits',{}).get('future_reserve_bytes')
        require(type(selected_future)is int and selected_future>0,'selected actual CPU slot future reserve required')
        require(request['remote_future_reserve_bytes']>=selected_future,'remote future reserve below selected actual manifest future reserve')
    observation,evidence=load_bound(request.get('root_terminal_evidence'),request_path.parent,request['max_evidence_bytes'],deadline)
    require(observation.get('schema')=='z1_horizon_terminal_observation_v1','root observation schema required')
    for key in ('policy','policy_id','scope','bank','job_id','source_root','private_root','guard_status','guard_returncode'):
        require(observation.get(key)==request.get(key),'root terminal observation mismatch: '+key)
    require(observation.get('slurm_terminal')is True,'root must observe terminal scheduler state')
    records=observation.get('records',{})
    terminal_name='terminal.json' if request['scope']=='training' else 'supervisor-terminal.json'
    require(terminal_name in records and 'intent.json' in records,'actual terminal and intent observation required')
    for name,item in records.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and isinstance(item,dict) and set(item)=={'sha256','value'},'bounded relative record hash/value required')
        require(re.fullmatch('[0-9a-f]{64}',item['sha256'])is not None,'record SHA malformed')
    shared=request.get('shared_source',{})
    require(shared.get('archive_sha256')==SHARED_SHA,'wrong shared source/archive identity')
    for role in ('mac_readback','vega_readback'):
        data,proof_bound=load_bound(shared.get(role),request_path.parent,request['max_evidence_bytes'],deadline)
        request['_bound_extra'].append(proof_bound)
        require(data.get('status')=='PASS' and data.get('archive_sha256')==SHARED_SHA and data.get('all_regular_file_hashes_verified')is True and data.get('exact_member_set')is True,'actual shared source readback required: '+role)
    if request['scope']=='training':
        require(isinstance(request.get('training_manifest_sha256'),str) and re.fullmatch('[0-9a-f]{64}',request['training_manifest_sha256']),'training manifest identity required')
    request['_bound_evidence']=evidence
    request['_bound_observation']={**evidence,'binding':{'records':records}}
    request['_terminal_records']=records
    if request['scope']=='training':request['training_attempt_kind']='original'  # no replacement route exists
    # The inherited transport reserves CONTROL_RESERVE plus the admitted future allocation.
    request['_future_reserve']=request['future_reserve_bytes']
    return request


def remote_program(deadline, maximum):
    paths=[Path(__file__).with_name('archive_contract.py'),Path(__file__).with_name('remote_policy_archive.py')]
    parts=[]
    for path in paths:
        check_deadline(deadline)
        require(path.is_file() and not path.is_symlink() and path.stat().st_size<=maximum,'bounded pinned helper source required')
        parts.append(path.read_text())
    return "import types,sys\nm=types.ModuleType('archive_contract')\nexec("+repr(parts[0])+",m.__dict__)\nsys.modules['archive_contract']=m\n"+parts[1].replace('from __future__ import annotations','')


def remote_request_payload(request,payload):
    remote=request.get('remote_future_reserve_bytes')
    require(type(remote)is int and remote>=REMOTE_FUTURE_MIN_BYTES,'remote_future_reserve_bytes must retain at least4563402752B')
    # The unchanged remote helper calls this field future_reserve_bytes. Its
    # inventory and every compressed stream write therefore retain Vega's gate.
    return {**payload,**{key:request.get(key)for key in('policy','private_root','bank','cpu_layout','scheduler_identity_capture')},
            'future_reserve_bytes':remote}


def call_remote(request, payload, *, action, output, stdout_path, deadline, reserve=CONTROL_RESERVE):
    check_deadline(deadline)
    payload=remote_request_payload(request,payload)
    reserve=CONTROL_RESERVE+request['future_reserve_bytes']
    remaining = deadline - time.monotonic()
    require(remaining > 5, "archive deadline expired before remote operation")
    program = remote_program(deadline, request["max_metadata_bytes"])
    payload = {**payload, "action": action, "remote_wall_seconds": max(1, min(560, int(remaining) - 2))}
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=2", "vega",
            shlex.join([request["remote_python"], "-B", "-u", "-c", program])]
    out_limit = request["max_archive_bytes"] if action == "stream" else request["max_metadata_bytes"]
    stderr_path = stdout_path.with_name(stdout_path.name + ".stderr.log")
    terminal_path = stdout_path.with_name(stdout_path.name + ".terminal.json")
    started = time.time()
    check_deadline(deadline)
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    counts = {"stdout": 0, "stderr": 0}
    status, error = "FAILED", None
    writer_errors = []
    writer = None
    try:
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        require(len(encoded) <= request["max_metadata_bytes"], "remote request exceeds metadata cap")

        def send_payload():
            try:
                process.stdin.write(encoded)
                process.stdin.flush()
                process.stdin.close()
            except BaseException as exc:
                writer_errors.append(exc)

        writer = threading.Thread(target=send_payload, name="policy-archive-ssh-input", daemon=True)
        writer.start()
        with stdout_path.open("xb", buffering=0) as stdout, stderr_path.open("xb", buffering=0) as stderr:
            while selector.get_map() or writer.is_alive():
                check_deadline(deadline)
                if writer_errors:
                    raise RuntimeError(f"SSH request input failed: {writer_errors[0]}")
                if not selector.get_map():
                    writer.join(timeout=min(.1, max(.01, deadline - time.monotonic())))
                    continue
                for key, _ in selector.select(timeout=min(1, max(.01, deadline - time.monotonic()))):
                    check_deadline(deadline)
                    data = os.read(key.fileobj.fileno(), 1024 * 1024)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    channel = key.data
                    counts[channel] += len(data)
                    require(counts[channel] <= (out_limit if channel == "stdout" else 1024 * 1024),
                            "remote response exceeded declared cap")
                    check_local_floor(output, len(data), reserve)
                    target = stdout if channel == "stdout" else stderr
                    view = memoryview(data)
                    while view:
                        check_deadline(deadline)
                        check_local_floor(output, len(view), reserve)
                        written = target.write(view)
                        require(written is not None and written > 0, "short streaming archive write")
                        view = view[written:]
                    os.fsync(target.fileno())
        writer.join(timeout=max(0, deadline - time.monotonic()))
        check_deadline(deadline)
        require(not writer.is_alive() and not writer_errors,
                "SSH request input did not finish before the deadline")
        process.wait(timeout=max(1, deadline - time.monotonic()))
        require(process.returncode == 0,
                f"remote {action} exited {process.returncode}; partial evidence retained")
        status = "TERMINAL0"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    error = (error or "") + "; SSH process did not reap within bounded cleanup"
        if writer is not None:
            writer.join(timeout=1)
        raise
    finally:
        selector.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        save_exclusive(terminal_path, json_bytes({
            "schema": "z1_horizon_multiroot_terminal_policy_archive_command_v1",
            "action": action, "status": status, "returncode": process.returncode,
            "started_local_epoch": started, "finished_local_epoch": time.time(),
            "error": error, "stream_bytes": counts, "argv": argv[:-1]+["<remote source bound by SHA256 below>"],
            "remote_program_sha256": hashlib.sha256(program.encode()).hexdigest(),
            "remote_files_written": 0}), reserve=reserve, deadline=deadline)


def verify_archive(archive_path, before, deadline):
    expected = before["nodes"]
    seen, origin_map = set(), {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            require(time.monotonic() < deadline, "deadline expired while checking archive members")
            name = member.name.rstrip("/")
            require(name in expected and name not in seen, "unexpected or duplicate archive member")
            require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                    "unsafe archive member path")
            seen.add(name)
            item = expected[name]
            require(member.mode == item["mode"] and member.uid == item["uid"] and member.gid == item["gid"]
                    and member.pax_headers.get("z1.mtime_ns") == str(item["mtime_ns"]),
                    "archive node metadata differs from source inventory")
            record = {key: item[key] for key in ("source_path", "type", "mode", "uid", "gid", "mtime_ns")}
            if item["type"] == "directory":
                require(member.isdir(), "directory member type differs")
            elif item["type"] == "symlink":
                require(member.issym() and member.linkname == item["target"], "symlink member target differs")
                record["target"] = item["target"]
            else:
                require(member.isfile() and member.size == item["bytes"], "regular member size/type differs")
                digest, size = hashlib.sha256(), 0
                with archive.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        require(time.monotonic() < deadline, "deadline expired while hashing archive member")
                        digest.update(block)
                        size += len(block)
                require(size == item["bytes"] and digest.hexdigest() == item["sha256"],
                        "regular archive member SHA differs from source inventory")
                record.update(bytes=size, sha256=digest.hexdigest())
            origin_map[name] = record
    require(seen == set(expected), "archive member set is incomplete")
    return origin_map


def read_archive_json(archive_path, member_name, deadline, maximum=2 * 1024**2):
    with tarfile.open(archive_path, "r:gz") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as exc:
            raise ValueError(f"required application record missing from tree: {member_name}") from exc
        require(member.isfile() and member.size <= maximum, "application record has invalid type/size")
        stream = archive.extractfile(member)
        chunks, size = [], 0
        while True:
            check_deadline(deadline)
            block = stream.read(64 * 1024)
            if not block:
                break
            size += len(block)
            require(size <= maximum, "application record exceeded parsing bound")
            chunks.append(block)
        return json.loads(b"".join(chunks))


def validate_archived_application_records(request, observation_binding, origin_map, archive_path, deadline, *, record_reader=None):
    reader = read_archive_json if record_reader is None else record_reader
    records=observation_binding['records']
    for relative,item in records.items():
        node=origin_map.get('policy-tree/'+relative)
        require(node is not None and node['type']=='file' and node['sha256']==item['sha256'],'observed terminal record hash differs from complete archive')
        actual=reader(archive_path,'policy-tree/'+relative,deadline,request['max_evidence_bytes'])
        require(actual==item['value'],'observed terminal record value differs from archive')
    terminal_name='terminal.json' if request['scope']=='training' else 'supervisor-terminal.json'
    terminal=records[terminal_name]['value'];intent=records['intent.json']['value']
    require(terminal.get('status')==request['guard_status'] and terminal.get('returncode')==request['guard_returncode'],'terminal status/returncode differs')
    success=terminal.get('status')==('COMPLETE' if request['scope']=='training' else 'RAW_REVIEW_REQUIRED')
    require(success or terminal.get('status')=='FAILED_NO_RETRY','unsupported terminal status')
    require(not success or terminal.get('returncode')==0,'success requires zero returncode')
    job=(terminal.get('allocation',{}).get('slurm_job_id') if request['scope']=='training' else terminal.get('allocation',{}).get('SLURM_JOB_ID'))
    require(job==request['job_id'],'terminal allocation differs from actual Slurm job')
    name=request['policy_id'];arm=request['policy']['arm'];seed=request['policy']['training_seed']
    if request['scope']=='training':
        cell=dict(name=name,arm=arm,seed=seed)
        require(intent.get('cell')==cell and intent.get('pair_context',{}).get('allocation',{}).get('slurm_job_id')==job,'training intent policy/job differs')
        if success:
            for filename in ('train-complete.json','training-config-intent.json','checkpoint-499-live.json'):
                require(filename in records,'successful training terminal observation missing '+filename)
            complete=records['train-complete.json']['value'];cfg=records['training-config-intent.json']['value'];live=records['checkpoint-499-live.json']['value']
            require(complete.get('status')=='COMPLETE' and complete.get('cell')==cell and complete.get('completed_updates')==500,'training completion identity/updates differ')
            require(cfg.get('cell')==cell and cfg.get('task')==request['policy']['task'] and cfg.get('source_manifest')==request['training_manifest_sha256'] and cfg.get('resume')is False and cfg.get('horizon_s')==request['policy']['native_window_s'] and cfg.get('velocity_pmax')==.5 and cfg.get('impulse_pmax')==.2,'training config identity/recipe differs')
            require(live.get('iteration')==499 and live.get('common_step_counter')==12000 and live.get('last_transition_index')==11999 and live.get('velocity_pmax')==.5 and live.get('impulse_pmax')==.2 and live.get('scheduler')is False,'training live clock/recipe differs')
            checkpoint=origin_map.get('policy-tree/train/model_499.pt',{})
            require(checkpoint.get('sha256')==live.get('checkpoint_sha256')==complete.get('checkpoint_sha256',{}).get('model_499.pt'),'final checkpoint hash differs')
    else:
        require(intent.get('policy')==request['policy'] and intent.get('scope')==request['scope'],'CPU intent policy/scope differs')
        require((None if request['scope']=='own_cpu32' else intent.get('evaluation_seed'))==request.get('bank'),'CPU scientific bank differs')
        require(intent.get('output_root')==request['source_root'],'CPU output root differs')
        require(intent.get('supervisor_context',{}).get('release_sha256')==request['cpu_supervisor_release']['sha256'],'CPU intent differs from actual supervisor release')
        if success:
            require('worker-result.json' in records,'successful CPU terminal lacks actual worker result')
            result=records['worker-result.json']['value']
            require(result.get('status')=='RAW_REVIEW_REQUIRED' and result.get('attempts')==(32 if request['scope']=='own_cpu32' else 160),'CPU raw result population/status differs')
    require(success or not terminal.get('status')=='COMPLETE','failure must not manufacture completion')


def consumer_receipt(request,origin_map,archive_path,before,after):
    def members(prefix):
        return {name[len(prefix)+1:]:item['sha256'] for name,item in origin_map.items() if name.startswith(prefix+'/') and item['type']=='file'}
    def digest(mapping):return hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    run=members('policy-tree');private=members('private-tree')
    return dict(schema='z1_horizon_terminal_multiroot_archive_receipt_v1',status='PASS',independent_readback='PASS',
        archive_only=True,readback_platform=sys.platform,readback_hostname=os.uname().nodename,technical_qualification=False,scientific_qualification=False,policy=request['policy'],scope=request['scope'],bank=request.get('bank'),
        run_root=request['source_root'],private_root=request['private_root'],members=run,tree_sha256=digest(run),
        private_members=private,private_tree_sha256=digest(private),training_manifest_sha256=request.get('training_manifest_sha256'),
        archive_path=str(archive_path),archive_sha256=sha_file(archive_path),job_before=before['job'],job_after=after['job'],
        shared_source=request['shared_source'],node_metadata=origin_map,raw_tree_extracted=False)


def _run_impl(request_path, output):
    operation_start = time.monotonic()
    hard_deadline = operation_start + MAX_WALL_SECONDS
    request_path = Path(os.path.abspath(request_path))
    output = Path(os.path.abspath(output))
    if os.path.lexists(output):
        require(not output.is_symlink(), "exclusive output path must not be a symlink")
    request_info = request_path.lstat()
    require(stat.S_ISREG(request_info.st_mode), "request must be a regular nonsymlink file")
    require(request_info.st_size <= 16 * 1024**2, "root request exceeds hard16MiB parse bound")
    request_path = request_path.resolve(strict=True)
    request_value = read_local_json(request_path, 16 * 1024**2, hard_deadline)
    require(type(request_value.get("wall_seconds")) is int and 0 < request_value["wall_seconds"] <= MAX_WALL_SECONDS,
            "declared local monotonic deadline must be1..600s")
    deadline = min(hard_deadline, operation_start + request_value["wall_seconds"])
    check_deadline(deadline)
    request_sha = sha_file(request_path, deadline)
    request = validate(request_value, request_path, deadline)
    require(request_info.st_size <= request["max_metadata_bytes"],
            "root request exceeds declared metadata cap")
    require(sha_file(request_path, deadline) == request_sha, "request changed during validation")
    helper_paths = sorted(Path(__file__).parent.glob("*.py"))
    helper_hashes = {path.name: sha_file(path, deadline) for path in helper_paths}
    require(not output.exists() and not output.is_symlink(), "new exclusive output directory required")
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    control_cap = 16 * request["max_metadata_bytes"] + 2 * request["max_evidence_bytes"] + 8 * 1024**2
    check_local_floor(parent, request["max_archive_bytes"] + control_cap, CONTROL_RESERVE + request['future_reserve_bytes'])
    output.parent.mkdir(parents=True, exist_ok=True)
    check_local_floor(output.parent, request["max_archive_bytes"] + control_cap, CONTROL_RESERVE + request['future_reserve_bytes'])
    output.mkdir()
    os.nice(19)

    # Remove the in-memory convenience field from the canonical copied request.
    canonical_request = {key: value for key, value in request.items() if not key.startswith("_")}
    canonical_request_bytes = json_bytes(canonical_request)
    require(len(canonical_request_bytes) <= request["max_metadata_bytes"],
            "canonical root request exceeds metadata cap")
    save_exclusive(output / 'root-request.json',canonical_request_bytes,reserve=CONTROL_RESERVE+request['future_reserve_bytes'],deadline=deadline)
    evidence = request["_bound_evidence"]
    copied_evidence = output / ("root-terminal-evidence" + Path(evidence["path"]).suffix)
    copy_bound_file(evidence["path"], copied_evidence, evidence["sha256"], evidence["bytes"], deadline, CONTROL_RESERVE+request["future_reserve_bytes"])
    observation_evidence = request["_bound_observation"]
    copied_observation = output / ("root-terminal-observation" + Path(observation_evidence["path"]).suffix)
    copy_bound_file(observation_evidence["path"], copied_observation,
                    observation_evidence["sha256"], observation_evidence["bytes"], deadline, CONTROL_RESERVE+request["future_reserve_bytes"])

    payload = {key: request[key] for key in (
        "campaign_root", "quota_root", "source_root", "scope", "policy_id", "job_id",
        "guard_returncode", "guard_status", "max_source_bytes", "max_archive_bytes",
        "max_members", "max_metadata_bytes")}
    if request["scope"] == "training":
        payload["training_attempt_kind"] = request["training_attempt_kind"]
    before_path = output / "remote-before.json"
    call_remote(request, payload, action="inspect", output=output, stdout_path=before_path,
                deadline=deadline)
    before = read_local_json(before_path, request["max_metadata_bytes"], deadline)
    require(before.get("scope") == request["scope"] and before.get("policy_id") == request["policy_id"],
            "remote inventory scope/policy differs")
    require(before.get("private_root") == request["private_root"] and before.get("requested_source_root") == request["source_root"],
            "inventoried request path differs from the approved policy/scope root")
    require(before.get("training_attempt_kind") == request.get("training_attempt_kind"),
            "inventoried training attempt kind differs")
    require(before.get("guard_returncode") == request.get("guard_returncode")
            and before.get("guard_status") == request.get("guard_status"),
            "remote inventory guard binding differs")
    mac_free_before_archive = check_local_floor(output, request["max_archive_bytes"], CONTROL_RESERVE + request['future_reserve_bytes'])

    archive_path = output / "terminal-policy-full.tar.gz"
    call_remote(request, {**payload, "before": before}, action="stream", output=output,
                stdout_path=archive_path, deadline=deadline)
    after_path = output / "remote-after.json"
    call_remote(request, payload, action="inspect", output=output, stdout_path=after_path,
                deadline=deadline)
    after = read_local_json(after_path, request["max_metadata_bytes"], deadline)
    require(before["nodes"] == after["nodes"], "terminal source tree changed during backup")
    require(after.get("scope") == request["scope"] and after.get("policy_id") == request["policy_id"]
            and after.get("private_root") == request["private_root"] and after.get("requested_source_root") == request["source_root"]
            and after.get("training_attempt_kind") == request.get("training_attempt_kind")
            and after.get("guard_returncode") == request.get("guard_returncode")
            and after.get("guard_status") == request.get("guard_status"),
            "after-inventory policy/scope/path/guard binding differs")
    for proof in (before["job"], after["job"]):
        require(proof["job_id"] == request["job_id"] and proof["terminal"] is True,
                "Slurm terminal proof does not bind the declared job")
    require((before["job"]["state"], before["job"]["exit_code"])
            == (after["job"]["state"], after["job"]["exit_code"]),
            "Slurm terminal state changed during archive")

    origin_map = verify_archive(archive_path, before, deadline)
    observation_binding = request["_bound_observation"]["binding"]
    validate_archived_application_records(request, observation_binding, origin_map, archive_path, deadline)
    origin_path = output / "canonical-origin-file-hash-map.json"
    origin_record = {
        "schema": "z1_horizon_multiroot_terminal_policy_origin_file_hash_map_v1",
        "scope": request["scope"], "policy_id": request["policy_id"],
        "training_attempt_kind": request.get("training_attempt_kind"),
        "source_root": before["requested_source_root"], "canonical_source_root": before["source_root"],
        "source_inventory_sha256": sha_file(before_path, deadline),
        "archive_sha256": sha_file(archive_path, deadline), "archive_member_count": len(origin_map),
        "regular_file_count": sum(item["type"] == "file" for item in origin_map.values()),
        "files_and_nodes": origin_map,
        "all_regular_files_have_sha256": all(item["type"] != "file" or len(item["sha256"]) == 64
                                               for item in origin_map.values()),
        "symlinks_are_metadata_only_and_never_followed": True,
    }
    origin_bytes = json_bytes(origin_record)
    require(len(origin_bytes) <= request["max_metadata_bytes"], "origin file map exceeds metadata cap")
    save_exclusive(origin_path,origin_bytes,reserve=CONTROL_RESERVE+request['future_reserve_bytes'],deadline=deadline)

    helper_hashes_after = {path.name: sha_file(path, deadline) for path in helper_paths}
    require(helper_hashes == helper_hashes_after, "helper source changed while archiving")
    require(sha_file(request_path, deadline) == request_sha, "request changed while archiving")
    require(sha_file(Path(evidence["path"]), deadline) == evidence["sha256"],
            "root terminal evidence changed while archiving")
    require(sha_file(Path(observation_evidence["path"]), deadline) == observation_evidence["sha256"],
            "root terminal observation changed while archiving")
    for artifact in request['_bound_extra']:
        require(sha_file(artifact['path'],deadline)==artifact['sha256'],'bound layout/shared proof changed during archive')
    require(time.monotonic() < deadline, 'deadline expired before receipt')
    archive_size = archive_path.stat().st_size
    require(archive_size <= request["max_archive_bytes"], "archive exceeds declared finite cap")
    receipt = {
        "schema": "z1_horizon_multiroot_terminal_policy_archive_verification_v1",
        "status": "PASS_ARCHIVE_ONLY_SINGLE_TERMINAL_POLICY",
        "scope": request["scope"], "policy_id": request["policy_id"],
        "root_request": {"path": str(request_path), "sha256": request_sha,
                         "copy_sha256": sha_file(output / "root-request.json", deadline)},
        "helper_sources_sha256": helper_hashes,
        "source_root": before["requested_source_root"],
        "canonical_source_root": before["source_root"], "source_unchanged": True,
        "guard_returncode": request["guard_returncode"], "guard_status": request["guard_status"],
        "training_attempt_kind": request.get("training_attempt_kind"),
        "root_terminal_evidence": {"source_path": evidence["path"], "sha256": evidence["sha256"],
                                   "copy_path": str(copied_evidence), "copy_sha256": sha_file(copied_evidence, deadline)},
        "root_terminal_observation": {"source_path": observation_evidence["path"],
                                      "sha256": observation_evidence["sha256"],
                                      "copy_path": str(copied_observation),
                                      "copy_sha256": sha_file(copied_observation, deadline)},
        "application_record_binding": observation_binding,
        "job_before": before["job"], "job_after": after["job"],
        "remote_before_path": str(before_path), "remote_before_sha256": sha_file(before_path, deadline),
        "remote_after_path": str(after_path), "remote_after_sha256": sha_file(after_path, deadline),
        "archive_path": str(archive_path), "archive_bytes": archive_size,
        "archive_sha256": sha_file(archive_path, deadline), "max_archive_bytes": request["max_archive_bytes"],
        "origin_hash_map_path": str(origin_path), "origin_hash_map_sha256": sha_file(origin_path, deadline),
        "archive_member_count": len(origin_map), "source_regular_bytes": before["regular_bytes"],
        "source_regular_file_count": origin_record["regular_file_count"],
        "exact_archive_member_set_and_all_regular_hashes_verified": True,
        "exact_member_set": True, "all_regular_file_hashes_verified": True,
        "full_tree_including_logs_caches_partials_extras_traces": True, "external_private_tree_included": True,
        "raw_tree_extracted_locally": False, "local_raw_duplicate_created": False,
        "preservation_floor_bytes": FLOOR, "control_reserve_bytes": CONTROL_RESERVE,
        "mac_future_reserve_bytes": request['future_reserve_bytes'],
        "remote_future_reserve_bytes": request['remote_future_reserve_bytes'],
        "mac_free_before_archive": mac_free_before_archive,
        "mac_free_after_archive": check_local_floor(output, reserve=CONTROL_RESERVE + request['future_reserve_bytes']),
        "remote_capacity_before": before["capacity"], "remote_capacity_after": after["capacity"],
        "remote_files_written": 0, "scientific_execution": False,
        "analysis_input_published": False, "new_seed_claims": False,
        "headroom_is_sampled_not_atomic_reservation": True,
        "scope_statement": "Exactly one declared terminal policy output and external private cache; no qualification inferred.",
        "verified_local_epoch": time.time(),
    }
    receipt['scope_statement']='Exactly one new horizon policy output plus its private cache; archive acceptance only.'
    normalized=consumer_receipt(request,origin_map,archive_path,before,after)
    normalized_bytes=json_bytes(normalized)
    require(len(normalized_bytes)<=request['max_metadata_bytes'],'consumer receipt exceeds metadata cap')
    save_exclusive(output/'consumer-backup-receipt.json',normalized_bytes,reserve=CONTROL_RESERVE+request['future_reserve_bytes'],deadline=deadline)
    receipt['consumer_receipt_sha256']=sha_file(output/'consumer-backup-receipt.json',deadline)
    receipt_bytes = json_bytes(receipt)
    require(len(receipt_bytes) <= request["max_metadata_bytes"], "archive receipt exceeds metadata cap")
    receipt_path = output / "archive-only-verification.json"
    save_exclusive(receipt_path,receipt_bytes,reserve=CONTROL_RESERVE+request['future_reserve_bytes'],deadline=deadline)
    return {"status": receipt["status"], "receipt_path": str(receipt_path),
            "receipt_sha256": sha_file(receipt_path, deadline), "archive_sha256": receipt["archive_sha256"],
            "origin_hash_map_sha256": receipt["origin_hash_map_sha256"]}


def _archive_watchdog(_signum, _frame):
    raise TimeoutError("terminal multiroot archive overall watchdog expired")


def run(request_path, output):
    # The signal watchdog bounds blocking local filesystem calls as well as
    # cooperative loops; the monotonic deadline remains the primary protocol.
    previous = signal.signal(signal.SIGALRM, _archive_watchdog)
    started = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, MAX_WALL_SECONDS)
    try:
        request_path = Path(os.path.abspath(request_path))
        info = request_path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_size <= 16 * 1024**2,
                "root request must be bounded regular input")
        value = read_local_json(request_path, 16 * 1024**2, started + MAX_WALL_SECONDS)
        require(type(value.get("wall_seconds")) is int and 0 < value["wall_seconds"] <= MAX_WALL_SECONDS,
                "declared local monotonic deadline must be1..600s")
        signal.setitimer(signal.ITIMER_REAL, max(.001, min(MAX_WALL_SECONDS,
                                                          value["wall_seconds"] - (time.monotonic() - started))))
        return _run_impl(request_path, output)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.request, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
