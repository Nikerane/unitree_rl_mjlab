"""Bounded metadata/raw-data own32 sealer. Never evaluates a policy."""
from __future__ import annotations
import argparse
import array
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import types
import zipfile

CONSUMER_SHA256='f4b40175786581063848c8b8d9b40651f203f9670418fcbe6d3da7274f3af449'
CAPACITY=881;WORLDS=32;DT=.00025
VELOCITY_LIMIT=3.1415;LOAD_CAPS=(.369,.246,.738,.369,.246,.0164)
WINDOWS={'legacy_0_50ms_half_open':(0,200),'inclusive_0_50ms':(0,201),
         'legacy_0_200ms_half_open':(0,800),'inclusive_50_200ms':(200,801),
         'authoritative_0_200ms_inclusive':(0,801)}
MAX_TRACE_BYTES=WORLDS*CAPACITY*(3*6*8+9)+WORLDS*8+CAPACITY*8+7*4096
IDENTITY_KEYS=('policy','runtime_identity','source_identity','observer_identity')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def tree_digest(members):return hashlib.sha256(json.dumps(members,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def inventory(root):
    root=Path(root)
    if not root.is_absolute()or not root.is_dir()or root.is_symlink():raise ValueError('actual absolute raw root required')
    result={}
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('raw symlink forbidden')
        if path.is_file():result[str(path.relative_to(root))]=sha(path)
    if not result:raise ValueError('empty raw tree')
    return result


def bound(reference,root=None):
    if not isinstance(reference,dict)or not {'path','sha256'}.issubset(reference):raise ValueError('actual hashed artifact required')
    path=Path(reference['path'])
    if not path.is_absolute()or path.is_symlink()or sha(path)!=reference['sha256']:raise ValueError('artifact path/hash differs')
    if root is not None and not path.resolve().is_relative_to(Path(root).resolve()):raise ValueError('raw producer reference escaped own raw tree')
    return path


def json_bound(reference,root=None):return json.loads(bound(reference,root).read_text())


def read_npy(payload):
    if payload[:6]!=b'\x93NUMPY'or len(payload)<10:raise ValueError('invalid NPY member')
    version=tuple(payload[6:8])
    if version==(1,0):offset=10;length=struct.unpack('<H',payload[8:10])[0]
    elif version in((2,0),(3,0)):
        if len(payload)<12:raise ValueError('truncated NPY header')
        offset=12;length=struct.unpack('<I',payload[8:12])[0]
    else:raise ValueError('unsupported NPY version')
    if length>4096 or offset+length>len(payload):raise ValueError('bounded NPY header required')
    try:header=ast.literal_eval(payload[offset:offset+length].decode('latin1').strip())
    except(Exception)as error:raise ValueError('malformed NPY header')from error
    if set(header)!={'descr','fortran_order','shape'}or header['fortran_order']is not False:raise ValueError('native C-order numeric arrays required')
    shape=header['shape'];dtype=header['descr'];formats={'<f4':('f',4),'<f8':('d',8),'<i8':('q',8),'|b1':('B',1)}
    if dtype not in formats or not isinstance(shape,tuple)or any(type(n)is not int or n<0 for n in shape):raise ValueError('unapproved NPY dtype/shape')
    count=math.prod(shape);kind,size=formats[dtype];raw=payload[offset+length:]
    if count*size!=len(raw)or len(raw)>MAX_TRACE_BYTES:raise ValueError('NPY shape/byte length mismatch')
    values=array.array(kind);values.frombytes(raw)
    if sys.byteorder!='little'and size>1:values.byteswap()
    if dtype=='|b1'and any(value not in(0,1)for value in values):raise ValueError('invalid native boolean')
    return dict(shape=shape,dtype=dtype,values=values)


def read_trace(path):
    with zipfile.ZipFile(path)as archive:
        entries=archive.infolist();names=[entry.filename for entry in entries]
        allowed={'qvel.npy','rolling.npy','pulse.npy','sample_index.npy','native_boundary.npy','valid_counts.npy','relative_time_s.npy'}
        if len(names)!=len(set(names))or not set(names).issubset(allowed)or sum(e.file_size for e in entries)>MAX_TRACE_BYTES:
            raise ValueError('unexpected/unbounded NPZ members')
        result={name[:-4]:read_npy(archive.read(name))for name in names}
    required={'sample_index':((32,881),'<i8'),'native_boundary':((32,881),'|b1'),
              'valid_counts':((32,),'<i8'),'relative_time_s':((881,),'<f8')}
    for name,(shape,dtype)in required.items():
        if name not in result or result[name]['shape']!=shape or result[name]['dtype']!=dtype:raise ValueError('native dense trace structure mismatch')
    counts=result['valid_counts']['values']
    if any(not 0<=count<=CAPACITY for count in counts):raise ValueError('invalid dense valid counts')
    if any(counts)and not {'qvel','rolling','pulse'}.issubset(result):raise ValueError('missing native producers')
    for name in('qvel','rolling','pulse'):
        if name not in result:continue
        value=result[name]
        if value['shape']!=(32,881,6)or value['dtype']not in('<f4','<f8'):raise ValueError('native producer dimensions/dtype differ')
        for world,count in enumerate(counts):
            data=value['values'][world*881*6:(world*881+count)*6]
            if any(not math.isfinite(x)or(name!='qvel'and x<0)for x in data):raise ValueError('invalid captured native producer')
    if any(not math.isclose(t,index*DT,rel_tol=0,abs_tol=1e-15)for index,t in enumerate(result['relative_time_s']['values'])):
        raise ValueError('literal timestamp grid differs')
    return result


def near(actual,expected):
    if expected is None:return actual is None
    if type(expected)is int:return type(actual)is int and actual==expected
    return isinstance(actual,(int,float))and math.isfinite(actual)and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12)


def audit_window(trace,world,start,stop,record):
    if record.get('status')!='COMPLETE'or record.get('required_samples')!=stop-start:raise ValueError('literal endpoint summary missing')
    for channel,producer in(('velocity','qvel'),('raw_rolling_impulse','rolling'),('held_impulse','pulse')):
        reported=record.get('channels',{}).get(channel,{})
        for joint in range(6):
            values=[trace[producer]['values'][(world*881+slot)*6+joint]for slot in range(start,stop)]
            if producer=='qvel':values=[abs(value)for value in values]
            limit=VELOCITY_LIMIT if producer=='qvel'else LOAD_CAPS[joint]
            bad=[slot+start for slot,value in enumerate(values)if value>limit];first=bad[0]if bad else None
            expected=dict(peak=max(values),minimum_margin=min(limit-value for value in values),exceedance_samples=len(bad),
                sample_count_times_dt_s=len(bad)*DT,first_count_including_onset=None if first is None else first+1,
                first_time_from_contact_s=None if first is None else first*DT,
                first_absolute_sample_index=None if first is None else trace['sample_index']['values'][world*881+first])
            if any(not isinstance(reported.get(key),list)or len(reported[key])!=6 or not near(reported[key][joint],value)for key,value in expected.items()):
                raise ValueError('native dense/endpoint parity mismatch')


def classify_world(episode,observer,boundary,count):
    native=boundary.get('native',{})
    if not {'accepted_contact_sample','first_raw_contact_sample'}.issubset(observer):raise ValueError('contact sensor absence is not a measured miss')
    if (episode.get('complete')is not True or type(episode.get('steps'))is not int or episode['steps']<=0
        or observer.get('active')is not False or observer.get('whole_attempt',{}).get('complete')is not True
        or native.get('latched')is not True or not any(native.get('termination_terms',{}).values())):
        raise ValueError('unknown/incomplete first-episode lifecycle')
    if observer.get('accepted_contact_sample')is not None:
        if episode.get('hit')is not True or count<801:raise ValueError('contact observed without literal801 coverage')
        return 'CONTACT_COMPLETE'
    if observer.get('first_raw_contact_sample')is not None or episode.get('hit')is not False or count!=0:
        raise ValueError('raw/accepted contact classification unresolved')
    if episode.get('timeout')is True:
        if native.get('termination_terms',{}).get('time_out')is not True:raise ValueError('timeout proof differs')
        return 'NO_CONTACT_TIMEOUT'
    if episode.get('timeout')is not False:raise ValueError('unknown timeout classification')
    return 'NO_CONTACT_TERMINAL'


def keyed(rows,key):
    if not isinstance(rows,list)or len(rows)!=32:raise ValueError('all32 raw worlds required')
    if any(type(row.get(key))is not int for row in rows):raise ValueError('explicit integer world identifiers required')
    result={row[key]:row for row in rows}
    if set(result)!=set(range(32))or len(result)!=len(rows):raise ValueError('raw worlds duplicate/missing')
    return result


def audit_raw(intent,root):
    case=root/'original-route+1';episodes=keyed(json.loads((case/'episodes.json').read_text()),'world')
    observer=keyed(json.loads((case/'observer-records.json').read_text()),'world_id')
    boundaries=keyed(json.loads((case/'native-boundary-records.json').read_text()),'world_id')
    reset=json.loads((case/'reset-observed.json').read_text());first=json.loads((case/'first-step-observed.json').read_text())
    if reset.get('worlds')!=list(range(32))or first.get('worlds')!=list(range(32)):raise ValueError('actual reset/first-step lifecycle incomplete')
    starts={world:0 for world in range(32)}
    for path in sorted(case.glob('physical-start-step-*.json')):
        event=json.loads(path.read_text())
        for world in event['worlds']:
            if world not in starts:raise ValueError('extra physical world')
            starts[world]+=1
        if any(value!=1 for value in event.get('counts',{}).values()):raise ValueError('extra physical start')
    if any(value!=1 for value in starts.values()):raise ValueError('exactly one physical start per world required')
    result=json.loads((root/'worker-result.json').read_text());reports=result.get('reports',[])
    if len(reports)!=1 or reports[0].get('case_id')!='original-route+1'or reports[0].get('report_sha256')!=sha(case/'report.json'):
        raise ValueError('actual case report binding differs')
    report=json.loads((case/'report.json').read_text());trace_ref=report['native_onset_trace'];trace_path=bound(trace_ref,root)
    trace=read_trace(trace_path);endpoint_ref=report['safety_endpoint_summary'];endpoints=json_bound(endpoint_ref,root)
    if endpoints.get('velocity_limit_rad_s')!=VELOCITY_LIMIT or endpoints.get('load_caps_nms')!=list(LOAD_CAPS):raise ValueError('native safety thresholds differ')
    endpoint_rows=keyed(endpoints['worlds'],'world_id');counts=trace['valid_counts']['values'];classifications=[]
    for world in range(32):
        episode=episodes[world];obs=observer[world];native=boundaries[world]['native'];endpoint=endpoint_rows[world];count=counts[world]
        if episode.get('episode')!=0:raise ValueError('engineering extras are not qualification first episodes')
        for row in(episode,obs,boundaries[world]):
            if bound(row['native_onset_trace'],root)!=trace_path or row['native_onset_trace']['sha256']!=trace_ref['sha256']:
                raise ValueError('producer linkage differs')
            if bound(row['safety_endpoint_summary'],root)!=bound(endpoint_ref,root):raise ValueError('endpoint linkage differs')
            mapping=json_bound(row['observer_mapping'],root)
            if not mapping:raise ValueError('actual mapping provenance missing')
        if obs.get('observed_samples')!=episode.get('steps',0)*80 or obs.get('whole_attempt',{}).get('samples')!=obs.get('observed_samples'):
            raise ValueError('full first-episode physics count/lifecycle differs')
        classification=classify_world(episode,obs,boundaries[world],count)
        if endpoint.get('valid_count')!=count:raise ValueError('endpoint valid count differs from trace')
        native_slots=[slot for slot in range(881)if trace['native_boundary']['values'][world*881+slot]]
        if endpoint.get('native_boundary_slots_zero_based')!=native_slots:raise ValueError('native boundary dense linkage differs')
        if classification=='CONTACT_COMPLETE':
            onset=obs['accepted_contact_sample'];samples=trace['sample_index']['values'][world*881:world*881+count]
            if type(onset)is not int or onset<=0 or samples[0]!=onset or samples[-1]!=obs['observed_samples'] or any(b-a!=1 for a,b in zip(samples,samples[1:])):raise ValueError('dense sample gap/onset mismatch')
            ticks=round(intent['policy']['native_window_s']/DT);native_count=native.get('contact_samples')
            if type(native_count)is not int or not ticks+1<=native_count<=ticks+80 or native_slots!=[native_count-1]or native_count>count:
                raise ValueError('actual native policy boundary coverage differs')
            if native.get('observed_samples')!=samples[native_count-1]:raise ValueError('native timestamp differs')
            for label,required_count in(('50ms',201),('200ms',801)):
                literal=obs.get('literal_depth',{}).get(label,{})
                if literal.get('complete')is not True or literal.get('count_including_onset')!=required_count or literal.get('sample')!=samples[required_count-1]:
                    raise ValueError('literal depth endpoint absent')
            for name,(start,stop)in WINDOWS.items():audit_window(trace,world,start,stop,endpoint['windows'][name])
            endpoint_class='COMPLETE_801'
        else:
            if native_slots or native.get('contact_samples')!=0 or endpoint['windows']['authoritative_0_200ms_inclusive'].get('status')!='UNKNOWN_MISSING_ENDPOINT':
                raise ValueError('no-contact episode contains fabricated contact endpoint')
            if any(endpoint['windows'][name].get('status')!='UNKNOWN_MISSING_ENDPOINT'or endpoint['windows'][name].get('channels')is not None for name in WINDOWS):
                raise ValueError('no-contact windows were zero-filled or invented')
            endpoint_class='NOT_APPLICABLE_NO_CONTACT'
        classifications.append(dict(world_id=world,classification=classification,contact_endpoint=endpoint_class,
            timeout=episode['timeout'],utility_incomplete=episode.get('incomplete'),termination_reasons=obs.get('end_reason')))
    return classifications


def load_consumer():
    path=Path(__file__).resolve().parent.parent/'package'/'supervisor'/'cpu_supervisor.py'
    if sha(path)!=CONSUMER_SHA256:raise ValueError('reviewed consumer contract changed')
    module=types.ModuleType('sealed_cpu_consumer');module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)  # Execute pinned source, never unchecked bytecode.
    return module


def build(request):
    required={'schema','intent','raw_root','raw_review','runtime_review','parity_review','independent_backup','seal_output'}
    if not isinstance(request,dict)or set(request)!=required or request.get('schema')!='z1_own_cpu32_seal_request_v1':raise ValueError('complete actual own32 seal request required')
    root=Path(request['raw_root']).resolve(strict=True);intent_path=bound(request['intent'],root);intent=json.loads(intent_path.read_text())
    if (intent_path!=root/'intent.json'or intent.get('scope')!='own_cpu32'or intent.get('attempts')!=32
        or intent.get('status')!='ADMITTED_ONCE'or intent.get('cases')!=['original-route+1']):raise ValueError('exact actual own32 intent required')
    output=Path(request['seal_output'])
    if not output.is_absolute()or output.exists()or output.resolve().is_relative_to(root):raise ValueError('new seal must be outside immutable raw tree')
    if json.loads((root/'actual-runtime.json').read_text())!=intent['runtime_identity']:raise ValueError('actual software runtime differs')
    training=json.loads((root/'training-audit.json').read_text())
    if(training.get('status')!='PASS'or training.get('finite_checkpoint')is not True or training.get('policy')!=intent['policy']
       or training.get('checkpoint_sha256')!=intent['policy']['checkpoint_sha256']or training.get('iteration')!=499
       or training.get('common_step_counter')!=12000 or training.get('last_transition_index')!=11999):
        raise ValueError('actual supervised checkpoint audit incomplete')
    members=inventory(root);digest=tree_digest(members);reviews={}
    for kind in('raw','runtime','parity'):
        reference=request[kind+'_review'];path=bound(reference)
        if path.resolve().is_relative_to(root):raise ValueError('reviews must stay outside immutable raw tree')
        review=json.loads(path.read_text())
        if (review.get('status')!='PASS'or review.get('review_kind')!=kind or review.get('raw_tree_sha256')!=digest
            or review.get('intent_sha256')!=request['intent']['sha256']or any(review.get(key)!=intent.get(key)for key in IDENTITY_KEYS)
            or not review.get('reviewer')):raise ValueError('actual identity-bound independent review required')
        reviews[kind]=review
    hardware_ref=dict(path=str(root/'hardware-prolog.json'),sha256=sha(root/'hardware-prolog.json'))
    if reviews['runtime'].get('hardware_prolog_sha256')!=hardware_ref['sha256']:raise ValueError('actual runtime review prolog mismatch')
    if reviews['parity'].get('native_endpoint_parity')!='PASS'or reviews['parity'].get('fixed_endpoint_parity')!='PASS':raise ValueError('actual native/literal endpoint parity review required')
    classification=audit_raw(intent,root)
    if reviews['raw'].get('world_classifications')!=classification:raise ValueError('reviewed raw world classifications differ')
    backup_path=bound(request['independent_backup'])
    if backup_path.resolve().is_relative_to(root):raise ValueError('backup receipt must stay outside raw tree')
    terminal_ref=dict(path=str(root/'supervisor-terminal.json'),sha256=sha(root/'supervisor-terminal.json'))
    seal=dict(schema='z1_own_cpu32_qualification_seal_v1',status='QUALIFIED',attempts=32,**{key:intent[key]for key in IDENTITY_KEYS},
        raw_review='PASS',runtime_review='PASS',parity_review='PASS',raw_root=str(root),raw_tree_sha256=digest,
        supervisor_terminal=terminal_ref,hardware_prolog=hardware_ref,independent_backup=request['independent_backup'],
        review_references={key:request[key+'_review']for key in reviews},world_classifications=classification,
        denominators=dict(all_first_episodes=32,contact_801=sum(row['classification']=='CONTACT_COMPLETE'for row in classification),
                          genuine_no_contact=sum(row['classification']!='CONTACT_COMPLETE'for row in classification)),
        unsafe_allowed_for_technical_qualification=True,scientific_safety_claim=False,consumer_sha256=CONSUMER_SHA256)
    consumer=load_consumer()
    # Exact unchanged consumer validation before publication via temporary seal
    # outside raw_root. No environment/torch import occurs in this consumer path.
    temporary=output.with_name(output.name+'.checking')
    with temporary.open('x')as stream:json.dump(seal,stream,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    reference=dict(path=str(temporary),sha256=sha(temporary))
    consumer.verify_qualification(intent,reference)
    if inventory(root)!=members:raise ValueError('raw tree changed during qualification')
    os.link(temporary,output);temporary.unlink()
    return dict(path=str(output),sha256=sha(output),status='QUALIFIED')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);args=parser.parse_args()
    print(json.dumps(build(json.loads(Path(args.request).read_text())),sort_keys=True))

if __name__=='__main__':main()
