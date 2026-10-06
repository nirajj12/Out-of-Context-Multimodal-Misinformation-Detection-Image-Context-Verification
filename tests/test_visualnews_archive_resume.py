"""Real small tar parsing with mocked HTTP; no network or model inference."""
import hashlib
import io
import json
import re
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.visualnews_archive_resume import ArchiveResumeError, HTTPRangeSource, ArchiveCursor, writer_lock
from scripts.visualnews_stream_embeddings import stream_resumable_image_embeddings, mapping_record
from tests.test_visualnews_stream_embeddings import targets, vectors, store, jpeg_bytes


class MockHTTP:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []
        self.responses = []
        self.overrides = {}
        self.transform = None
        self.truncate = False

    def get(self, url, headers, **kwargs):
        match = re.fullmatch(r'bytes=(\d+)-(\d*)', headers['Range'])
        start = int(match[1]); end = int(match[2]) if match[2] else len(self.payload)-1
        self.requests.append((start, end, headers.copy()))
        body = self.payload[start:end+1]
        if self.transform:
            body = self.transform(start, end, body)
        if self.truncate and not match[2]:
            body = body[:512]
        raw = io.BytesIO(body)
        response = SimpleNamespace(status_code=206, url=url, raw=raw, headers={
            'Content-Range': f'bytes {start}-{end}/{len(self.payload)}',
            'Content-Length': str(end-start+1), 'ETag': '"synthetic-v1"',
            'Last-Modified': 'Sun, 17 Apr 2022 19:54:29 GMT', 'Date': 'Wed, 07 Oct 2026 00:00:00 GMT'})
        response.headers.update(self.overrides.get('headers', {}))
        response.status_code = self.overrides.get('status', 206)
        response.read_count = 0
        response.close = lambda: setattr(response, 'read_count', raw.tell())
        self.responses.append(response)
        return response

    def source(self):
        return HTTPRangeSource('https://synthetic.invalid/origin.tar', self.get)


def tar_bytes(table, format=tarfile.PAX_FORMAT, pax=None, corrupt=None, unrelated=False):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w', format=format, pax_headers=pax) as archive:
        if unrelated:
            info=tarfile.TarInfo('../escape.jpg');info.size=1
            archive.addfile(info,io.BytesIO(b'x'))
        for i, row in enumerate(table.to_dict('records')):
            data=b'broken' if i==corrupt else jpeg_bytes()
            info=tarfile.TarInfo(row['image_path'].removeprefix('visual_news/'))
            info.size=len(data)
            if format==tarfile.PAX_FORMAT:
                info.pax_headers={'mtime':'1.123456789', 'comment': 'local extension'}
            archive.addfile(info,io.BytesIO(data))
    return output.getvalue()


def run(s, http, **kwargs):
    return stream_resumable_image_embeddings(s, lambda image: np.asarray(image),
                                            lambda items: vectors(len(items)), source=http.source(), gpu_batch_size=2, **kwargs)


def hashes(s):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in s.chunk_dir.iterdir()}


def expected_ends(payload):
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:') as archive:
        return [info.offset_data+((info.size+511)//512)*512 for info in archive if info.isfile()]


def test_initial_stop_multiple_absolute_resumes_and_noop(tmp_path):
    table=targets(7);http=MockHTTP(tar_bytes(table));ends=expected_ends(http.payload)
    s=store(tmp_path,table);r=run(s,http,max_new_images_per_session=1)
    assert r['status']=='stopped_intentionally' and r['encoded_this_run']==2
    assert r['committed_archive_offset']==ends[1] and r['resume_start_offset']==0
    before=hashes(s)
    http.requests.clear()
    s=store(tmp_path,table);r=run(s,http,max_new_images_per_session=2)
    assert r['resume_start_offset']==ends[1] and r['committed_archive_offset']==ends[3]
    assert http.requests[0][:2]==(0,511)
    assert http.requests[1][0]==ends[1] and http.requests[1][1]==len(http.payload)-1
    assert http.requests[1][2]['If-Range']=='"synthetic-v1"'
    assert all(hashes(s)[name]==value for name,value in before.items())
    s=store(tmp_path,table);r=run(s,http)
    assert r['status']=='complete' and r['encoded_this_run']==3 and s.completed_ids==set(range(7))
    maps=[__import__('pandas').read_parquet(s.chunk_paths(i)[1]) for i in range(len(s.chunks))]
    assert __import__('pandas').concat(maps).metadata_id.is_unique
    http.requests.clear()
    r=stream_resumable_image_embeddings(store(tmp_path,table), None, lambda _:pytest.fail('GPU'),source=http.source())
    assert not http.requests and r['encoded_this_run']==0


@pytest.mark.parametrize('format',[tarfile.GNU_FORMAT,tarfile.PAX_FORMAT])
def test_long_names_and_global_pax_resume(tmp_path,format):
    table=targets(5)
    table['image_path']=[p.rsplit('/',1)[0]+'/'+('x'*150)+f'{i}.jpg' for i,p in enumerate(table.image_path)]
    http=MockHTTP(tar_bytes(table,format,pax={'uid':'17','comment':'global'} if format==tarfile.PAX_FORMAT else None))
    ends=expected_ends(http.payload)
    s=store(tmp_path,table);r=run(s,http,max_new_images_per_session=2)
    assert r['committed_archive_offset']==ends[1]
    if format==tarfile.PAX_FORMAT:
        saved=json.loads((tmp_path/'image_archive_cursor.json').read_text())
        assert saved['pax_headers']=={'uid':'17','comment':'global'}
        with http.source().open_tar(saved['next_offset'],saved['pax_headers'],saved['source']) as archive:
            following=next(iter(archive))
            assert following.uid==17 and following.name==table.iloc[2].image_path.removeprefix('visual_news/')
    r=run(store(tmp_path,table),http,max_new_images_per_session=2)
    assert r['resume_start_offset']==ends[1] and r['committed_archive_offset']==ends[3]
    assert run(store(tmp_path,table),http)['status']=='complete'


def test_interrupted_gpu_batch_does_not_advance_cursor(tmp_path):
    table=targets(5);http=MockHTTP(tar_bytes(table));s=store(tmp_path,table)
    calls=[]
    def interrupted(items):
        calls.append(len(items))
        if len(calls)==2:raise KeyboardInterrupt()
        return vectors(len(items))
    with pytest.raises(KeyboardInterrupt):
        stream_resumable_image_embeddings(s,lambda image:np.asarray(image),interrupted,source=http.source(),gpu_batch_size=2)
    assert s.completed_ids=={0,1}
    assert json.loads((tmp_path/'image_archive_cursor.json').read_text())['next_offset']==expected_ends(http.payload)[1]
    r=run(store(tmp_path,table),http)
    assert r['encoded_this_run']==3 and r['status']=='complete'
    assert not s.failures_path.exists()


def test_checkpoint_before_cursor_crash_harmless_replay(tmp_path,monkeypatch):
    table=targets(5);http=MockHTTP(tar_bytes(table));s=store(tmp_path,table)
    original=ArchiveCursor.commit
    def crash(self,offset,*args,**kwargs):
        if offset>0:raise KeyboardInterrupt()
        return original(self,offset,*args,**kwargs)
    monkeypatch.setattr(ArchiveCursor,'commit',crash)
    with pytest.raises(KeyboardInterrupt):run(s,http)
    assert s.completed_ids=={0,1}
    assert json.loads((tmp_path/'image_archive_cursor.json').read_text())['next_offset']==0
    before=hashes(s)
    monkeypatch.setattr(ArchiveCursor,'commit',original)
    assert run(store(tmp_path,table),http)['encoded_this_run']==3
    assert all(hashes(s)[name]==value for name,value in before.items())


@pytest.mark.parametrize('overrides',[
    {'status':200}, {'headers':{'Content-Range':'bytes 0-511/999'}},
    {'headers':{'Content-Range':'garbage'}}, {'headers':{'ETag':'"changed"'}},
    {'headers':{'Last-Modified':'Mon, 18 Apr 2022 19:54:29 GMT'}},
    {'headers':{'Content-Encoding':'gzip'}}, {'headers':{'Content-Length':'1'}},
])
def test_resume_bad_http_headers_rejected_before_any_payload(tmp_path,overrides):
    table=targets(5);http=MockHTTP(tar_bytes(table));s=store(tmp_path,table)
    run(s,http,max_new_images_per_session=2);before=hashes(s)
    http.overrides=overrides
    with pytest.raises(ArchiveResumeError):run(store(tmp_path,table),http)
    assert http.responses[-1].read_count==0 and hashes(s)==before


def test_bad_nonzero_range_even_when_probe_succeeds(tmp_path):
    table=targets(5);http=MockHTTP(tar_bytes(table));run(store(tmp_path,table),http,max_new_images_per_session=2)
    original=http.get
    def broken(url,headers,**kwargs):
        response=original(url,headers,**kwargs)
        if headers['Range']!='bytes=0-511':response.status_code=200
        return response
    source=HTTPRangeSource('https://synthetic.invalid/origin.tar',broken)
    with pytest.raises(ArchiveResumeError,match='HTTP 200'):
        stream_resumable_image_embeddings(store(tmp_path,table),None,None,source=source)
    assert http.responses[-1].read_count==0


def test_truncated_transport_is_not_decode_failure(tmp_path):
    table=targets(5);http=MockHTTP(tar_bytes(table));http.truncate=True;s=store(tmp_path,table)
    with pytest.raises(ArchiveResumeError,match='truncated'):run(s,http)
    assert not s.failures_path.exists() and not s.completed_paths
    assert json.loads((tmp_path/'image_archive_cursor.json').read_text())['next_offset']==0


def test_earlier_decode_failure_retry_bounded_after_forward_cursor(tmp_path):
    table=targets(5);http=MockHTTP(tar_bytes(table));failed_once=[False]
    # Source bytes/identity never change: simulate one transient Pillow/preprocessing failure.
    def preprocess(image):
        if not failed_once[0]:
            failed_once[0]=True
            raise ValueError('transient decoder problem')
        return np.asarray(image)
    s=store(tmp_path,table)
    r=stream_resumable_image_embeddings(s,preprocess,lambda items:vectors(len(items)),source=http.source(),gpu_batch_size=2,max_new_images_per_session=2)
    assert r['unresolved_failures']==1 and r['committed_archive_offset']==expected_ends(http.payload)[2]
    journal=json.loads((tmp_path/'image_archive_failures.jsonl').read_text())
    http.requests.clear()
    r=run(store(tmp_path,table),http)
    assert r['status']=='complete' and r['encoded_this_run']==3 and r['unresolved_failures']==0
    assert http.requests[1][:2]==(journal['data_offset'],journal['data_offset']+journal['size']-1)
    assert http.requests[2][0]==expected_ends(http.payload)[2]


def test_decode_failure_eof_retry_does_not_open_forward_archive(tmp_path):
    table=targets(5);http=MockHTTP(tar_bytes(table,corrupt=0));s=store(tmp_path,table)
    r=run(s,http)
    assert r['status']=='incomplete' and r['remaining_targets']==1
    assert json.loads((tmp_path/'image_archive_cursor.json').read_text())['archive_eof']
    http.requests.clear();r=run(store(tmp_path,table),http)
    assert r['status']=='incomplete' and r['encoded_this_run']==0 and r['failed_this_run']==1
    assert len(http.requests)==2 and http.requests[1][1]-http.requests[1][0]+1==6


def test_legacy_5000_row_chunk_reused_without_cursor(tmp_path):
    table=targets(5002);s=store(tmp_path,table,chunk_size=5000)
    s.add([mapping_record(r) for r in table.iloc[:5000].to_dict('records')],vectors(5000))
    before=hashes(s)
    http=MockHTTP(tar_bytes(table))
    # Legacy recovery starts at zero, but never recomputes the 5000 completed IDs.
    s=store(tmp_path,table,chunk_size=1000);r=run(s,http)
    assert r['resume_start_offset']==0 and r['encoded_this_run']==2 and r['status']=='complete'
    assert r['scanned_members_this_run']==5002
    assert [c['validation']['rows'] for c in s.chunks]==[5000,2]
    assert all(hashes(s)[name]==value for name,value in before.items())


def test_conflicting_writer_and_progress_change_stop(tmp_path):
    table=targets(2);s=store(tmp_path,table)
    with writer_lock(s.state_path):
        with pytest.raises(ArchiveResumeError,match='writer lock'):store(tmp_path,table)
        with pytest.raises(ArchiveResumeError,match='writer lock'):run(s,MockHTTP(tar_bytes(table)))
    state=json.loads(s.state_path.read_text());state['external']='changed';s.state_path.write_text(json.dumps(state))
    with pytest.raises(ArchiveResumeError,match='another writer'):s.add([mapping_record(table.iloc[0])],vectors(1))


def test_cursor_failure_journal_corruption_rejected(tmp_path):
    table=targets(5);http=MockHTTP(tar_bytes(table,corrupt=0));s=store(tmp_path,table);run(s,http)
    path=tmp_path/'image_archive_failures.jsonl';path.write_text(path.read_text().replace('broken','changed'))
    # Error text contains Pillow text, so guarantee a prefix byte modification.
    data=path.read_bytes();path.write_bytes(data.replace(b'UnidentifiedImageError',b'UnidentifiedImageErroX'))
    with pytest.raises(ArchiveResumeError,match='checksum'):run(store(tmp_path,table),http)


@pytest.mark.parametrize('kind',['checksum','single_zero','dangling_pax','sparse','global_size','unknown_type','compressed'])
def test_malformed_or_unsupported_tar_stops_without_skipping(tmp_path,kind):
    table=targets(3)
    output=io.BytesIO()
    if kind in ('checksum','single_zero'):
        payload=tar_bytes(table.iloc[:1])
        end=expected_ends(payload)[0]
        if kind=='checksum':payload=payload[:end]+b'x'*512+payload[end+512:]
        else:payload=payload[:end]+bytes(512)+b'x'*512+payload[end+1024:]
    elif kind=='dangling_pax':
        payload=tar_bytes(table)
        # First 512-byte extended header + extension payload, followed by EOF.
        payload=payload[:1024]+bytes(8192)
    elif kind=='compressed':
        import gzip
        payload=gzip.compress(tar_bytes(table));payload+=bytes((-len(payload))%512)
    else:
        with tarfile.open(fileobj=output,mode='w',format=tarfile.PAX_FORMAT,
                          pax_headers={'size':'3'} if kind=='global_size' else None) as archive:
            info=tarfile.TarInfo(table.iloc[0].image_path.removeprefix('visual_news/'))
            if kind=='sparse':info.type=tarfile.GNUTYPE_SPARSE
            elif kind=='unknown_type':info.type=b'Z'
            archive.addfile(info)
        payload=output.getvalue()
    http=MockHTTP(payload);s=store(tmp_path,table)
    with pytest.raises(ArchiveResumeError):run(s,http)
    assert not s.completed_ids and not s.failures_path.exists()
    cursor=tmp_path/'image_archive_cursor.json'
    if cursor.exists():assert json.loads(cursor.read_text())['next_offset']==0


def test_pax_local_size_override_uses_real_logical_boundary(tmp_path):
    table=targets(3);payload=tar_bytes(table)
    # Rewrite ordinary header size to zero while a PAX size record gives true payload size.
    output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w',format=tarfile.PAX_FORMAT) as archive:
        for row in table.to_dict('records'):
            info=tarfile.TarInfo(row['image_path'].removeprefix('visual_news/'))
            data=jpeg_bytes();info.size=len(data);info.pax_headers={'size':str(len(data))}
            archive.addfile(info,io.BytesIO(data))
    payload=bytearray(output.getvalue())
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:') as archive:
        header_offsets=[info.offset_data-512 for info in archive]
    for offset in header_offsets:
        header=payload[offset:offset+512];header[124:136]=b'00000000000\0'
        header[148:156]=b'        '
        header[148:156]=f'{sum(header):06o}\0 '.encode()
        payload[offset:offset+512]=header
    payload=bytes(payload);ends=expected_ends(payload);http=MockHTTP(payload)
    first=run(store(tmp_path,table),http,max_new_images_per_session=2)
    assert first['committed_archive_offset']==ends[1]
    assert run(store(tmp_path,table),http)['status']=='complete'


def test_interrupt_before_first_batch_and_storage_failure_keep_old_cursor(tmp_path,monkeypatch):
    table=targets(3);http=MockHTTP(tar_bytes(table));s=store(tmp_path,table)
    count=[0]
    def interrupted(image):
        count[0]+=1
        if count[0]==2:raise KeyboardInterrupt()
        return np.asarray(image)
    with pytest.raises(KeyboardInterrupt):
        stream_resumable_image_embeddings(s,interrupted,None,source=http.source(),gpu_batch_size=2)
    assert not s.completed_ids and json.loads((tmp_path/'image_archive_cursor.json').read_text())['next_offset']==0
    s=store(tmp_path,table)
    def disk_full():raise OSError('storage full')
    monkeypatch.setattr(s,'flush',disk_full)
    with pytest.raises(OSError,match='storage full'):run(s,http)
    assert not s.failures_path.exists() and json.loads((tmp_path/'image_archive_cursor.json').read_text())['next_offset']==0
    assert run(store(tmp_path,table),http)['encoded_this_run']==3


def test_new_1000_row_checkpoints_max_gpu_64_and_short_tail(tmp_path):
    table=targets(1027);http=MockHTTP(tar_bytes(table));s=store(tmp_path,table,chunk_size=1000);batches=[]
    def encode(items):batches.append(len(items));return vectors(len(items))
    r=stream_resumable_image_embeddings(s,lambda image:np.asarray(image),encode,source=http.source(),gpu_batch_size=64)
    assert r['status']=='complete' and [c['validation']['rows'] for c in s.chunks]==[1000,27]
    assert max(batches)==64 and batches[-2:]==[40,27]
    assert r['encoder_seconds']>=0 and r['checkpoint_seconds']>0
    assert r['target_completion_percent']==100 and r['archive_scan_percent']<100


def test_missing_target_verified_eof_is_not_an_intentional_stop(tmp_path):
    table=targets(3);http=MockHTTP(tar_bytes(table.iloc[:2]));s=store(tmp_path,table)
    with pytest.raises(ArchiveResumeError,match='absent at verified archive EOF'):run(s,http)
    cursor=json.loads((tmp_path/'image_archive_cursor.json').read_text())
    assert cursor['archive_eof'] and cursor['next_offset']==expected_ends(http.payload)[1]
    count=len(http.requests)
    with pytest.raises(ArchiveResumeError,match='absent at verified archive EOF'):run(store(tmp_path,table),http)
    assert len(http.requests)==count+1  # Probe only; do not repeat a completed scan.


def test_identity_total_changed_and_retry_transport_failure_are_systemic(tmp_path):
    table=targets(5);http=MockHTTP(tar_bytes(table));s=store(tmp_path,table)
    run(s,http,max_new_images_per_session=2)
    http.payload+=bytes(512)
    with pytest.raises(ArchiveResumeError,match='identity changed'):run(store(tmp_path,table),http)
    assert http.responses[-1].read_count==0
    # A separate failed-image setup, then truncate just the bounded retry body.
    other=tmp_path/'other';s=store(other,table);http=MockHTTP(tar_bytes(table,corrupt=0));run(s,http)
    before=s.failures_path.read_bytes()
    http.transform=lambda start,end,body:body[:-1] if start>0 else body
    with pytest.raises(ArchiveResumeError,match='truncated'):run(store(other,table),http)
    assert s.failures_path.read_bytes()==before


def test_weak_etag_without_strong_date_is_rejected(tmp_path):
    table=targets(2);http=MockHTTP(tar_bytes(table))
    http.overrides={'headers':{'ETag':'W/"weak"','Date':'Sun, 17 Apr 2022 19:54:30 GMT'}}
    with pytest.raises(ArchiveResumeError,match='no usable strong'):run(store(tmp_path,table),http)
    assert http.responses[-1].read_count==0


def test_failure_rate_threshold_persistent_and_no_gpu(tmp_path):
    table=targets(5);output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w') as archive:
        for row in table.to_dict('records'):
            info=tarfile.TarInfo(row['image_path'].removeprefix('visual_news/'));info.size=6
            archive.addfile(info,io.BytesIO(b'broken'))
    s=store(tmp_path,table)
    with pytest.raises(RuntimeError,match='failure safety limit'):
        stream_resumable_image_embeddings(s,lambda _:None,lambda _:pytest.fail('GPU'),source=MockHTTP(output.getvalue()).source())
    assert len(s.failures_path.read_text().splitlines())==5 and not s.completed_ids


@pytest.mark.parametrize('invalid_size',['-1','abc'])
def test_invalid_pax_size_never_silently_becomes_zero(tmp_path,invalid_size):
    table=targets(3);output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w',format=tarfile.PAX_FORMAT) as archive:
        info=tarfile.TarInfo(table.iloc[0].image_path.removeprefix('visual_news/'))
        data=jpeg_bytes();info.size=len(data);info.pax_headers={'size':invalid_size}
        archive.addfile(info,io.BytesIO(data))
    s=store(tmp_path,table)
    with pytest.raises(ArchiveResumeError,match='invalid PAX size'):run(s,MockHTTP(output.getvalue()))
    assert not s.completed_ids and not s.failures_path.exists()


def test_gnu_longlink_before_target_does_not_shift_resume(tmp_path):
    table=targets(3);output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w',format=tarfile.GNU_FORMAT) as archive:
        link=tarfile.TarInfo('irrelevant-link');link.type=tarfile.SYMTYPE;link.linkname='x'*200
        archive.addfile(link)
        for row in table.to_dict('records'):
            data=jpeg_bytes();info=tarfile.TarInfo(row['image_path'].removeprefix('visual_news/'));info.size=len(data)
            archive.addfile(info,io.BytesIO(data))
    http=MockHTTP(output.getvalue());ends=expected_ends(http.payload)
    assert run(store(tmp_path,table),http,max_new_images_per_session=2)['committed_archive_offset']==ends[1]
    assert run(store(tmp_path,table),http)['status']=='complete'


def test_cursor_pending_vectors_and_changed_marker_rejected(tmp_path):
    table=targets(3);http=MockHTTP(tar_bytes(table));s=store(tmp_path,table)
    run(s,http,max_new_images_per_session=2)
    cursor=ArchiveCursor(s)
    s.add([mapping_record(table.iloc[2])],vectors(1))
    with pytest.raises(ArchiveResumeError,match='pending vectors'):
        cursor.commit(cursor.state['next_offset'],cursor.state['pax_headers'],cursor.state['source'])
    path=tmp_path/'image_archive_cursor.json';state=json.loads(path.read_text())
    state['checkpoints'][0]['embedding_sha256']='changed'
    from scripts.visualnews_archive_resume import state_checksum
    state['state_sha256']=state_checksum(state,'state_sha256');path.write_text(json.dumps(state))
    with pytest.raises(ArchiveResumeError,match='missing/changed checkpoint'):run(store(tmp_path,table),http)


def test_published_pair_before_marker_interruption_recovered_then_reused(tmp_path,monkeypatch):
    import scripts.visualnews_stream_embeddings as module
    table=targets(5);http=MockHTTP(tar_bytes(table));s=store(tmp_path,table)
    original=module.write_json_atomic
    def interrupted(path,value):
        if path.name=='image_chunk_00000.complete.json':raise KeyboardInterrupt()
        return original(path,value)
    monkeypatch.setattr(module,'write_json_atomic',interrupted)
    with pytest.raises(KeyboardInterrupt):run(s,http)
    array,mapping,marker=s.chunk_paths(0)
    assert array.exists() and mapping.exists() and not marker.exists()
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [array,mapping]}
    assert json.loads((tmp_path/'image_archive_cursor.json').read_text())['next_offset']==0
    monkeypatch.setattr(module,'write_json_atomic',original)
    reloaded=store(tmp_path,table)
    assert reloaded.completed_ids=={0,1} and marker.exists()
    assert run(reloaded,http)['encoded_this_run']==3
    assert all(hashes(reloaded)[name]==checksum for name,checksum in before.items())


@pytest.mark.parametrize('case',['wrong_start','wrong_end','wrong_total'])
def test_wrong_nonzero_content_range_is_rejected(tmp_path,case):
    table=targets(5);http=MockHTTP(tar_bytes(table));run(store(tmp_path,table),http,max_new_images_per_session=2)
    original=http.get
    def incorrect(url,headers,**kwargs):
        response=original(url,headers,**kwargs)
        if headers['Range']!='bytes=0-511':
            start,end,_=http.requests[-1];total=len(http.payload)
            if case=='wrong_start':start-=512
            elif case=='wrong_end':end-=512
            else:total+=512;end+=512
            response.headers['Content-Range']=f'bytes {start}-{end}/{total}'
            response.headers['Content-Length']=str(end-start+1)
        return response
    source=HTTPRangeSource('https://synthetic.invalid/origin.tar',incorrect)
    with pytest.raises(ArchiveResumeError):
        stream_resumable_image_embeddings(store(tmp_path,table),None,None,source=source)
    assert http.responses[-1].read_count==0


def test_notebook_checkpoint_session_settings_saved_safely():
    import ast
    from tests.test_visualnews_stream_embeddings import production_notebook_cells
    values={}
    for cell in production_notebook_cells():
        for node in ast.parse(cell.source).body:
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
                for name in node.targets:
                    if isinstance(name,ast.Name):values[name.id]=node.value.value
    assert values['GPU_BATCH_SIZE']==64 and values['CHECKPOINT_SIZE']==1000
    assert values['MAX_NEW_IMAGES_PER_SESSION'] is None
    assert all(values[name] is False for name in ['RUN_IMAGE_EMBEDDING','RUN_TEXT_EMBEDDING',
        'RUN_IMAGE_MERGE','RUN_TEXT_MERGE','RUN_FINAL_VALIDATION'])


def test_cursor_offset_corruption_and_uncommitted_journal_record_corruption_stop(tmp_path):
    table=targets(5);http=MockHTTP(tar_bytes(table));s=store(tmp_path,table);run(s,http,max_new_images_per_session=2)
    path=tmp_path/'image_archive_cursor.json';state=json.loads(path.read_text());state['next_offset']+=512
    path.write_text(json.dumps(state))
    with pytest.raises(ArchiveResumeError,match='cursor checksum'):run(store(tmp_path,table),http)
    assert not s.failures_path.exists()
    # Valid JSON in an uncommitted journal tail also needs record integrity.
    other=tmp_path/'other';s=store(other,table);http=MockHTTP(tar_bytes(table,corrupt=0))
    identity=http.source().probe();cursor=ArchiveCursor(s);cursor.commit(0,{},identity)
    cursor.record_failure(table.iloc[0].to_dict(),ValueError('test'),0,1536,6,True,identity)
    journal=other/'image_archive_failures.jsonl';record=json.loads(journal.read_text());record['data_offset']+=512
    journal.write_text(json.dumps(record)+'\n')
    with pytest.raises(ArchiveResumeError,match='malformed/conflicting failure journal'):run(store(other,table),http)
