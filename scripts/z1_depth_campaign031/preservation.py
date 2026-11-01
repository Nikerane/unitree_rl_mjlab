"""Independent archive member verification and a single finite evidence join."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from .files import canonical, digest, inventory, need, read_json, reference, regular_path, relative_name, safe_path, sync_dir, verify_ref, write_once
from .manifest import LIMITS


def stream_hash(stream, cap):
    sha=hashlib.sha256();size=0
    while True:
        block=stream.read(min(1024**2,cap-size+1))
        if not block:break
        size+=len(block);need(size<=cap,'stream byte cap exceeded');sha.update(block)
    return dict(sha256=sha.hexdigest(),size=size)


def archive_reference(path):
    path=regular_path(path)
    with path.open('rb') as stream:row=stream_hash(stream,512*1024**2)
    return dict(path=str(path),**row)


def archive(root, destination, *, identity, cap):
    root=safe_path(root);destination=safe_path(destination)
    members=inventory(root,cap)
    need('.depth031-members.json' not in members,'reserved archive member name consumed')
    embedded=canonical(dict(schema='z1_depth_archive_members031',identity=identity,members=members))
    # Exclusive archive is never overwritten/deleted on a partial failure.
    with destination.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w|',format=tarfile.PAX_FORMAT) as tf:
            info=tarfile.TarInfo('.depth031-members.json');info.size=len(embedded);info.mode=0o600;tf.addfile(info,io.BytesIO(embedded))
            for name,expected in members.items():
                path=regular_path(root/relative_name(name))
                info=tf.gettarinfo(str(path),arcname=name)
                need(info.isfile() and info.size==expected['size'],'member changed before archive')
                info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
                with path.open('rb') as src:tf.addfile(info,src)
        raw.flush();os.fsync(raw.fileno())
    sync_dir(destination.parent)
    need(inventory(root,cap)==members,'source members changed during preservation')
    receipt=dict(schema='z1_depth_archive031',identity=identity,source_root=str(root),cap_bytes=cap,
        members=members,embedded_manifest=dict(sha256=hashlib.sha256(embedded).hexdigest(),size=len(embedded)),archive=archive_reference(destination))
    verify_archive(receipt)
    return receipt


def verify_archive(receipt):
    need(receipt.get('schema')=='z1_depth_archive031','new031 archive receipt required')
    need(archive_reference(receipt['archive']['path'])==receipt['archive'],'archive SHA/size differs')
    actual={};total=0;embedded_seen=False
    with tarfile.open(receipt['archive']['path'],mode='r|') as tf:
        for item in tf:
            relative_name(item.name)
            if item.name=='.depth031-members.json':
                need(not embedded_seen and item.isfile() and item.size<=16*1024**2,'duplicate/invalid embedded member manifest')
                body=tf.extractfile(item).read(16*1024**2+1)
                need(dict(sha256=hashlib.sha256(body).hexdigest(),size=len(body))==receipt['embedded_manifest'],'embedded manifest bytes differ')
                need(body==canonical(dict(schema='z1_depth_archive_members031',identity=receipt['identity'],members=receipt['members'])),'embedded member manifest differs')
                embedded_seen=True;continue
            need(item.isfile() and item.name not in actual and item.name in receipt['members'],'extra/nonregular/duplicate archive member')
            expected=receipt['members'][item.name]
            need(item.size==expected['size'],'member size differs')
            actual[item.name]=stream_hash(tf.extractfile(item),item.size)
            total+=item.size;need(total<=receipt['cap_bytes'],'archive member cap exceeded')
    need(embedded_seen and actual==receipt['members'],'archive members differ')
    return dict(status='VERIFIED_MEMBERS',identity=receipt['identity'],archive_sha256=receipt['archive']['sha256'],members_sha256=digest(actual))


def final_join(manifest, training, evaluations, backups, *, downloads=None):
    from .join import final_join as join
    return join(manifest,training,evaluations,backups,downloads=downloads)


def verify_ssd(manifest,destination):
    """Adjacent read-only UUID/free check; receipts may live off exFAT."""
    import plistlib
    import subprocess
    from .manifest import BACKUP_ROOT,SSD_UUID
    destination=safe_path(destination)
    need(destination.is_relative_to(Path(BACKUP_ROOT)) and manifest['backup_root']==BACKUP_ROOT,'exact031 SSD destination required')
    result=subprocess.run(['diskutil','info','-plist','/Volumes/NikeraneSSD'],check=True,capture_output=True,timeout=10)
    info=plistlib.loads(result.stdout)
    need(info.get('VolumeUUID')==SSD_UUID,'SSD VolumeUUID differs')
    fs=os.statvfs('/Volumes/NikeraneSSD')
    need(fs.f_bavail*fs.f_frsize>=LIMITS['cap']+1024**3,'SSD independent preservation headroom insufficient')
    return dict(volume_uuid=SSD_UUID,free_bytes=fs.f_bavail*fs.f_frsize)
