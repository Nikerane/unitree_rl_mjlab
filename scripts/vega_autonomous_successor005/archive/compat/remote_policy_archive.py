"""Read-only Vega inventory/archive stream for one declared policy tree.

This module is sent as source to the explicitly selected Vega interpreter. It
does not create temporary files or write to the campaign filesystem.
"""
from __future__ import annotations
from archive_contract import validate_paths, CAMPAIGN, accounting_identity

import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time

FLOOR = 512 * 1024**2
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def raise_walk_error(error):
    raise error


def sha(path):
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def headroom(quota_root, future=0):
    quota = int(os.getxattr(quota_root, "ceph.quota.max_bytes"))
    used = int(os.getxattr(quota_root, "ceph.dir.rbytes"))
    filesystem = shutil.disk_usage(quota_root).free
    effective = min(quota - used, filesystem)
    require(effective >= FLOOR+32*1024**2+future, "remote Ceph quota/filesystem below512MiB floor")
    return {"quota_max_bytes": quota, "quota_used_bytes": used,
            "filesystem_free_bytes": filesystem, "effective_free_bytes": effective}


def job_proof(job_id, identity_capture):
    require(isinstance(job_id, str) and re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id),
            "exact Slurm allocation/task ID required")
    commands = {
        "sacct": ["sacct", "--allocations", "--parsable2", "--noheader", "--jobs", job_id,
                  "--format=JobID,JobIDRaw,State,ExitCode,Start,End,Elapsed,NodeList,AllocCPUS"],
        "squeue": ["squeue", "--noheader", "--jobs", job_id, "--format=%i|%T|%N"],
    }
    result = {"job_id": job_id, "captured_epoch": time.time()}
    for name, argv in commands.items():
        process = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        require(len(process.stdout.encode()) + len(process.stderr.encode()) <= 1024 * 1024,
                "scheduler response exceeded1MiB")
        result[name] = {"argv": argv, "returncode": process.returncode,
                        "stdout": process.stdout, "stderr": process.stderr}
    require(result["sacct"]["returncode"] == 0, "Slurm accounting unavailable")
    rows = [line.split("|") for line in result["sacct"]["stdout"].splitlines() if line.strip()]
    exact = [row for row in rows if len(row)==9 and row[1] == job_id]
    require(len(rows)==1 and len(exact) == 1, "exact allocation/task row missing or duplicated")
    row = exact[0]
    require(identity_capture.get("job_id")==job_id and identity_capture.get("accounting_job_id")==row[0],"explicit captured scheduler identity differs")
    identity = accounting_identity(job_id,row[0],row[1],identity_capture["capture"])
    state = row[2].split()[0].rstrip("+")
    require(state in TERMINAL and row[5] not in ("", "Unknown", "None"), "declared job is still live")
    queue = result["squeue"]
    require(not queue["stdout"].strip(), "declared terminal job remains queued/running")
    require(queue["returncode"] == 0 or (queue["returncode"] == 1 and
            "Invalid job id specified" in queue["stderr"]), "Slurm queue query failed")
    result.update(state=state, exit_code=row[3], terminal=True,identity=identity)
    return result


def validate_guard(request,job):
    status=request['guard_status'];code=request['guard_returncode']
    success=status==('COMPLETE' if request['scope']=='training' else 'RAW_REVIEW_REQUIRED')
    require(success or status=='FAILED_NO_RETRY','actual terminal status required')
    require(not success or code==0,'success requires zero process returncode')
    require(not success or (job['state']=='COMPLETED' and job['exit_code']=='0:0'),'process success contradicts Slurm terminal state')


def inventory(request):
    require(request['campaign_root']==CAMPAIGN,'foreign campaign root')
    roots=validate_paths(request)
    for path in roots.values():
        require(path.resolve(strict=True)==path and path.is_dir() and not path.is_symlink(),'canonical actual roots required; aliases forbidden')
    quota_root=Path(request['quota_root']).resolve(strict=True)
    capacity=headroom(quota_root,request['future_reserve_bytes'])
    job=job_proof(request['job_id'],request['scheduler_identity_capture']);validate_guard(request,job)
    nodes, total = {}, 0

    def add(path, member):
        nonlocal total
        require(member not in nodes and not Path(member).is_absolute() and ".." not in Path(member).parts,
                "duplicate or unsafe archive member")
        info = path.lstat()
        node = {"source_path": str(path), "mode": stat.S_IMODE(info.st_mode),
                "mtime_ns": info.st_mtime_ns, "uid": info.st_uid, "gid": info.st_gid}
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
            require(total <= request["max_source_bytes"], "single source tree exceeds source-byte cap")
            node.update(type="file", bytes=info.st_size, sha256=sha(path))
        elif stat.S_ISDIR(info.st_mode):
            node.update(type="directory")
        elif stat.S_ISLNK(info.st_mode):
            node.update(type="symlink", target=os.readlink(path))
        else:
            raise ValueError("special filesystem node found; refusing to omit it")
        nodes[member] = node
        require(len(nodes) <= request["max_members"], "single source tree exceeds member-count cap")

    for prefix,root in roots.items():
        add(root,prefix)
        for current,directories,files in os.walk(root,followlinks=False,onerror=raise_walk_error):
            directories.sort();files.sort();current=Path(current)
            for name in list(directories):
                child=current/name;add(child,prefix+'/'+child.relative_to(root).as_posix())
                if child.is_symlink():directories.remove(name)
            for name in files:
                child=current/name;add(child,prefix+'/'+child.relative_to(root).as_posix())
    return dict(schema='z1_horizon_terminal_multiroot_inventory_v1',scope=request['scope'],policy_id=request['policy_id'],
        campaign_root=CAMPAIGN,quota_root=str(quota_root),requested_source_root=request['source_root'],source_root=request['source_root'],
        private_root=request['private_root'],training_attempt_kind=request.get('training_attempt_kind'),guard_status=request['guard_status'],
        guard_returncode=request['guard_returncode'],job=job,capacity=capacity,nodes=nodes,regular_bytes=total,captured_epoch=time.time(),remote_files_written=0)


class BoundedOutput:
    def __init__(self, limit, quota_root, future):
        self.future=future
        self.limit = limit
        self.quota_root = quota_root
        self.count = 0

    def write(self, data):
        if not data:
            return 0
        require(self.count + len(data) <= self.limit, "archive exceeds finite compressed-byte cap")
        headroom(self.quota_root,self.future)
        sys.stdout.buffer.write(data)
        self.count += len(data)
        return len(data)

    def flush(self):
        sys.stdout.buffer.flush()


def stream(request):
    before = request['before']
    current=inventory(request)
    require(current['nodes']==before['nodes'] and current['job']['job_id']==before['job']['job_id'] and current['job']['state']==before['job']['state'] and current['job']['exit_code']==before['job']['exit_code'],'stream requires same admitted terminal two-root inventory')
    output = BoundedOutput(request["max_archive_bytes"], Path(request["quota_root"]),request["future_reserve_bytes"])
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0, compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for name, item in sorted(before["nodes"].items()):
                path = Path(item["source_path"])
                info = path.lstat()
                require(info.st_mtime_ns == item["mtime_ns"] and stat.S_IMODE(info.st_mode) == item["mode"]
                        and info.st_uid == item["uid"] and info.st_gid == item["gid"],
                        "source metadata changed before archive stream")
                member = tarfile.TarInfo(name)
                member.mode, member.mtime = item["mode"], info.st_mtime
                member.uid, member.gid = item["uid"], item["gid"]
                member.pax_headers = {"z1.mtime_ns": str(item["mtime_ns"])}
                if item["type"] == "file":
                    require(stat.S_ISREG(info.st_mode) and info.st_size == item["bytes"],
                            "source file changed type/size before stream")
                    member.size = item["bytes"]
                    source_fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    with os.fdopen(source_fd, "rb") as source:
                        archive.addfile(member, source)
                elif item["type"] == "directory":
                    require(stat.S_ISDIR(info.st_mode), "source directory changed type")
                    member.type = tarfile.DIRTYPE
                    archive.addfile(member)
                else:
                    require(stat.S_ISLNK(info.st_mode) and os.readlink(path) == item["target"],
                            "source symlink changed")
                    member.type, member.linkname = tarfile.SYMTYPE, item["target"]
                    archive.addfile(member)


def main():
    request = json.load(sys.stdin)
    seconds = request.get("remote_wall_seconds")
    require(type(seconds) is int and 0 < seconds <= 560, "bounded remote wall deadline required")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("remote archive deadline")))
    signal.alarm(seconds)
    os.nice(19)
    if request.get("action") == "inspect":
        result = inventory(request)
        encoded = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
        require(len(encoded) <= request["max_metadata_bytes"], "remote inventory exceeded metadata cap")
        sys.stdout.buffer.write(encoded)
    elif request.get("action") == "stream":
        stream(request)
    else:
        raise ValueError("unknown remote archive action")


if __name__ == "__main__":
    main()
